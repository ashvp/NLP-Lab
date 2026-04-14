#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LEN    5000   /* max input chars per document */
#define MAX_TOKENS 2000   /* max words extracted per document */
#define MAX_WORD   64     /* max chars per word */

/* ──────────────────────────────────────────────
   Tokenise: lowercase, strip punctuation, split
   ────────────────────────────────────────────── */
int tokenize(const char *text, char tokens[][MAX_WORD], int max_tokens) {
    int  count = 0;
    int  i     = 0;
    int  tlen  = (int)strlen(text);

    while (i < tlen && count < max_tokens) {
        /* skip non-alpha */
        while (i < tlen && !isalpha((unsigned char)text[i])) i++;
        if (i >= tlen) break;

        /* read word */
        int j = 0;
        while (i < tlen && isalpha((unsigned char)text[i]) && j < MAX_WORD - 1) {
            tokens[count][j++] = (char)tolower((unsigned char)text[i++]);
        }
        tokens[count][j] = '\0';
        if (j > 0) count++;
    }
    return count;
}

/* ──────────────────────────────────────────────
   Smith-Waterman on word token arrays
   match=+2  mismatch=-1  gap=-1
   ────────────────────────────────────────────── */
int smith_waterman_words(char tokens1[][MAX_WORD], int m,
                         char tokens2[][MAX_WORD], int n) {
    const int MATCH    =  2;
    const int MISMATCH = -1;
    const int GAP      = -1;

    /* Allocate score matrix on the heap */
    int **score = (int **)malloc((size_t)(m + 1) * sizeof(int *));
    if (!score) return 0;
    for (int i = 0; i <= m; i++) {
        score[i] = (int *)calloc((size_t)(n + 1), sizeof(int));
        if (!score[i]) { /* out of memory – free and bail */
            for (int k = 0; k < i; k++) free(score[k]);
            free(score);
            return 0;
        }
    }

    int max_score = 0;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = score[i-1][j-1] +
                       (strcmp(tokens1[i-1], tokens2[j-1]) == 0 ? MATCH : MISMATCH);
            int up   = score[i-1][j] + GAP;
            int left = score[i][j-1] + GAP;

            int best = 0;
            if (diag > best) best = diag;
            if (up   > best) best = up;
            if (left > best) best = left;

            score[i][j] = best;
            if (best > max_score) max_score = best;
        }
    }

    for (int i = 0; i <= m; i++) free(score[i]);
    free(score);

    return max_score;
}

int main(void) {
    char *text1 = (char *)malloc(MAX_LEN * sizeof(char));
    char *text2 = (char *)malloc(MAX_LEN * sizeof(char));
    if (!text1 || !text2) return 1;

    if (fgets(text1, MAX_LEN, stdin) == NULL) { free(text1); free(text2); return 0; }
    text1[strcspn(text1, "\n")] = '\0';

    if (fgets(text2, MAX_LEN, stdin) == NULL) { free(text1); free(text2); return 0; }
    text2[strcspn(text2, "\n")] = '\0';

    /* Tokenise both documents */
    char (*tokens1)[MAX_WORD] = malloc(MAX_TOKENS * MAX_WORD * sizeof(char));
    char (*tokens2)[MAX_WORD] = malloc(MAX_TOKENS * MAX_WORD * sizeof(char));
    if (!tokens1 || !tokens2) { free(text1); free(text2); return 1; }

    int m = tokenize(text1, tokens1, MAX_TOKENS);
    int n = tokenize(text2, tokens2, MAX_TOKENS);

    int score = smith_waterman_words(tokens1, m, tokens2, n);
    printf("\nLocal Alignment Score: %d\n", score);

    free(text1); free(text2);
    free(tokens1); free(tokens2);
    return 0;
}