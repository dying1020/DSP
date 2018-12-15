#include "hmm.h"

int main(int argc, char *argv[])
{
	if (argc != 2 + 1) {
		printf("Wrong argument format\n");
		printf("Usage: ./calc_acc result.txt testing_answer.txt\n");
		exit(1);
	}

	const char *result_file = argv[1];
	const char *ans_file = argv[2];

    double correct_num = 0, total_num = 0;
	char pred[MAX_LINE] = "";
	char ans[MAX_LINE] = "";
    double likelihood;

    FILE *fp1, *fp2;
    fp1 = open_or_die(result_file, "r");
    fp2 = open_or_die(ans_file, "r");
	while (fscanf(fp1, "%s %le", pred, &likelihood) > 0 && fscanf(fp2, "%s", ans) > 0) {
        if (strcmp(pred, ans) == 0) {
            correct_num++;
        }
        total_num++;
    }
    fclose(fp1);
    fclose(fp2);
    printf("correct: %.0f\n", correct_num);
    printf("total: %.0f\n", total_num);
    printf("accuracy: %f\n", correct_num / total_num);

    return 0;
}