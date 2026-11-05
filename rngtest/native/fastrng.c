/*
 * Высокопроизводительные ядра статистических тестов (C99).
 *
 * Собирается в разделяемую библиотеку и вызывается из Python через ctypes
 * (см. rngtest/native/__init__.py). При компиляции с -O3 компилятор
 * автоматически векторизует циклы (SSE/AVX на x86-64, NEON на ARM).
 *
 * Входные последовательности:
 *   - "unpacked": массив uint8 со значениями 0/1 (по биту на байт);
 *   - "packed":   массив uint8, по 8 бит в байте, старший бит первым.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Алгоритм Берлекэмпа–Мэсси: линейная сложность блока длины n.       */
int bm_linear_complexity(const uint8_t *s, int n)
{
    uint8_t *C = (uint8_t *)calloc((size_t)n + 1, 1);
    uint8_t *B = (uint8_t *)calloc((size_t)n + 1, 1);
    uint8_t *T = (uint8_t *)malloc((size_t)n + 1);
    int L = 0, m = -1, i, j;
    C[0] = B[0] = 1;
    for (i = 0; i < n; i++) {
        uint8_t d = s[i];
        for (j = 1; j <= L; j++) d ^= C[j] & s[i - j];
        if (d) {
            memcpy(T, C, (size_t)n + 1);
            int shift = i - m;
            for (j = 0; j + shift <= n; j++) C[j + shift] ^= B[j];
            if (L <= i / 2) {
                L = i + 1 - L;
                m = i;
                memcpy(B, T, (size_t)n + 1);
            }
        }
    }
    free(C); free(B); free(T);
    return L;
}

/* Линейная сложность для N подряд идущих блоков длины M. */
void lc_blocks(const uint8_t *bits, int N, int M, int32_t *out)
{
    for (int b = 0; b < N; b++) out[b] = bm_linear_complexity(bits + (size_t)b * M, M);
}

/* ------------------------------------------------------------------ */
/* Операции над упакованными битами (в 8 раз меньше памяти).           */

static int popcnt8(uint8_t x) { return __builtin_popcount((unsigned)x); }

/* Число единиц. */
int64_t packed_popcount(const uint8_t *p, int64_t nbits)
{
    int64_t full = nbits / 8, cnt = 0, i;
    const uint64_t *w = (const uint64_t *)p;
    int64_t nw = full / 8;
    for (i = 0; i < nw; i++) cnt += __builtin_popcountll(w[i]);
    for (i = nw * 8; i < full; i++) cnt += popcnt8(p[i]);
    int rem = (int)(nbits % 8);
    if (rem) cnt += popcnt8((uint8_t)(p[full] >> (8 - rem)));
    return cnt;
}

static inline int getbit(const uint8_t *p, int64_t i) { return (p[i >> 3] >> (7 - (i & 7))) & 1; }

/* Число несовпадений s_i != s_{i+d}, i = 0..n-d-1 (автокорреляция / серии при d=1). */
int64_t packed_autocorr(const uint8_t *p, int64_t nbits, int64_t d)
{
    int64_t cnt = 0, i = 0, lim = nbits - d;
    if (d % 8 == 0) {
        /* Сдвиг на целое число байт: XOR байтов + popcount. */
        int64_t db = d / 8, nb = lim / 8;
        for (i = 0; i < nb; i++) cnt += popcnt8(p[i] ^ p[i + db]);
        i = nb * 8;
    } else {
        /* Общий случай: собираем сдвинутое 8-битное окно из двух байт. */
        int64_t db = d >> 3, sh = d & 7, nb = (lim / 8);
        if (nb > 0 && (nb + db + 1) * 8 > nbits + 7) nb--;
        for (int64_t k = 0; k < nb; k++) {
            uint8_t a = p[k];
            uint8_t b = (uint8_t)((p[k + db] << sh) | (p[k + db + 1] >> (8 - sh)));
            cnt += popcnt8(a ^ b);
        }
        i = nb * 8;
    }
    for (; i < lim; i++) cnt += getbit(p, i) ^ getbit(p, i + d);
    return cnt;
}

/* Наибольшая серия единиц в каждом из N блоков длины M (unpacked). */
void longest_runs(const uint8_t *bits, int64_t N, int64_t M, int32_t *out)
{
    for (int64_t b = 0; b < N; b++) {
        const uint8_t *s = bits + b * M;
        int32_t cur = 0, best = 0;
        for (int64_t i = 0; i < M; i++) {
            cur = s[i] ? cur + 1 : 0;
            if (cur > best) best = cur;
        }
        out[b] = best;
    }
}

/* Упаковка 0/1 -> байты (старший бит первым). */
void pack_bits(const uint8_t *bits, int64_t nbits, uint8_t *out)
{
    int64_t nb = (nbits + 7) / 8;
    memset(out, 0, (size_t)nb);
    for (int64_t i = 0; i < nbits; i++) out[i >> 3] |= (uint8_t)(bits[i] << (7 - (i & 7)));
}
