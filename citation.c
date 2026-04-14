#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TEXT 1000000   /* FIX: 1 MB — handles full research papers */

/*
 * Detects three citation styles:
 *   1. Numeric bracket:      [1]  [12]  [123]
 *   2. Author-year bracket:  [Vaswani et al., 2017]  [LeCun 1989]
 *   3. Parenthetical year:   (2017)  (1986)   – 4-digit year inside parens
 */
int detect_citations(const char *text) {
    int count = 0;
    int len   = (int)strlen(text);

    for (int i = 0; i < len; i++) {

        /* ── Style 1 & 2: [...] ── */
        if (text[i] == '[') {
            int j = i + 1;

            /* Skip whitespace */
            while (j < len && isspace((unsigned char)text[j])) j++;

            if (j < len) {
                if (isdigit((unsigned char)text[j])) {
                    /* Style 1: numeric — [1], [12], [123] */
                    while (j < len && isdigit((unsigned char)text[j])) j++;
                    /* Allow optional comma-separated list: [1, 2, 3] */
                    while (j < len && text[j] != ']' && (j - i) < 40) j++;
                    if (j < len && text[j] == ']') {
                        count++;
                        i = j; /* advance past closing bracket */
                    }
                } else if (isupper((unsigned char)text[j])) {
                    /* Style 2: author-year — [Vaswani et al., 2017] */
                    int k = j;
                    while (k < len && text[k] != ']' && (k - i) < 60) k++;
                    if (k < len && text[k] == ']') {
                        /* Must contain at least one digit (the year) */
                        int has_digit = 0;
                        for (int m = j; m < k; m++) {
                            if (isdigit((unsigned char)text[m])) { has_digit = 1; break; }
                        }
                        if (has_digit) {
                            count++;
                            i = k;
                        }
                    }
                }
            }
        }

        /* ── Style 3: parenthetical year — (2017) ── */
        if (text[i] == '(') {
            int j = i + 1;
            while (j < len && isspace((unsigned char)text[j])) j++;
            /* Look for a 4-digit year 1900–2099 */
            if (j + 3 < len &&
                isdigit((unsigned char)text[j])   &&
                isdigit((unsigned char)text[j+1]) &&
                isdigit((unsigned char)text[j+2]) &&
                isdigit((unsigned char)text[j+3])) {

                int year = (text[j]   - '0') * 1000 +
                           (text[j+1] - '0') * 100  +
                           (text[j+2] - '0') * 10   +
                           (text[j+3] - '0');

                if (year >= 1900 && year <= 2099) {
                    int k = j + 4;
                    while (k < len && isspace((unsigned char)text[k])) k++;
                    if (k < len && text[k] == ')') {
                        count++;
                        i = k;
                    }
                }
            }
        }
    }

    return count;
}

int main(void) {
    char *text = (char *)malloc(MAX_TEXT * sizeof(char));
    if (!text) return 1;

    size_t len = 0;
    int    c;

    while ((c = getchar()) != EOF && len < (size_t)(MAX_TEXT - 1)) {
        text[len++] = (char)c;
    }
    text[len] = '\0';

    int citations = detect_citations(text);
    printf("\nCitations detected: %d\n", citations);

    free(text);
    return 0;
}