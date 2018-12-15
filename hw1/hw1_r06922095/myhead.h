#ifndef MY_HEADER_
#define MY_HEADER_

#include "hmm.h"

#ifndef MAX_TEST_LINE
    #define MAX_TEST_LINE 5000
#endif

#ifndef MAX_TRAIN_LINE
    #define MAX_TRAIN_LINE 15000
#endif

typedef struct {
	int seq_num;
	int seq[MAX_SEQ];
} Observation;

typedef struct {
	int seq_num;
	int state_num;
	double table[MAX_SEQ][MAX_STATE];
} Table;

typedef struct {
    int seq_num;
    int state_num;
    double table[MAX_SEQ][MAX_STATE][MAX_STATE];
} Epsilon;

/**
 * @param array of observation
 * @param filename
 * @return number of observation
 */
int get_data(Observation *observs, const char *filename)
{
	int i = 0, j, index;
	FILE *fp = open_or_die(filename, "r");

	char token[MAX_LINE] = "";
	while (fscanf(fp, "%s", token) > 0) {
		if (token[0] == '\0' || token[0] == '\n') {
			continue;
		}
		
		for (j = 0; j < MAX_LINE; j++) {
			switch (token[j]) {
				case 'A':
					index = 0;
					break;
				case 'B':
					index = 1;
					break;
				case 'C':
					index = 2;
					break;
				case 'D':
					index = 3;
					break;
				case 'E':
					index = 4;
					break;
				case 'F':
					index = 5;
					break;
				case '\0':
				default:
					index = -1;
					break;
			}

			if (index < 0) {
				observs[i].seq_num = j;
				i++;
				break;
			}
			observs[i].seq[j] = index;
		}
	}
	fclose(fp);
	return i;
}

#endif