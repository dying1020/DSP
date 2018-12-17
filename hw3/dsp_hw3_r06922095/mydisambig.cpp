#include <stdio.h>
#include "Ngram.h"
#include "VocabMap.h"

#ifndef MAX_CANDIDATE
    #define MAX_CANDIDATE 1050 // Maximum number of candidate for one zhuyin is 1014
#endif

// Get P(W1) -- uigram
LogP getUigramProb(Vocab &voc, Ngram &lm, const char *w1)
{
    VocabIndex wid1 = voc.getIndex(w1);

    if(wid1 == Vocab_None)  //OOV
        wid1 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = { Vocab_None };
    return lm.wordProb( wid1, context);
}

// Get P(W2 | W1) -- bigram
LogP getBigramProb(Vocab &voc, Ngram &lm, const char *w1, const char *w2)
{
    VocabIndex wid1 = voc.getIndex(w1);
    VocabIndex wid2 = voc.getIndex(w2);

    if(wid1 == Vocab_None)  //OOV
        wid1 = voc.getIndex(Vocab_Unknown);
    if(wid2 == Vocab_None)  //OOV
        wid2 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = { wid1, Vocab_None };
    return lm.wordProb( wid2, context);
}

// Get P(W3 | W1, W2) -- trigram
LogP getTrigramProb(Vocab &voc, Ngram &lm, const char *w1, const char *w2, const char *w3) 
{
    VocabIndex wid1 = voc.getIndex(w1);
    VocabIndex wid2 = voc.getIndex(w2);
    VocabIndex wid3 = voc.getIndex(w3);

    if(wid1 == Vocab_None)  //OOV
        wid1 = voc.getIndex(Vocab_Unknown);
    if(wid2 == Vocab_None)  //OOV
        wid2 = voc.getIndex(Vocab_Unknown);
    if(wid3 == Vocab_None)  //OOV
        wid3 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = { wid2, wid1, Vocab_None };
    return lm.wordProb( wid3, context );
}

/*** Constant in Vocab.h
const VocabIndex	Vocab_None = (VocabIndex)-1;
const VocabString	Vocab_Unknown = "<unk>";
const VocabString	Vocab_SentStart = "<s>";
const VocabString	Vocab_SentEnd = "</s>";
*/

int main(int argc, char *argv[])
{
    if (argc != 8 + 1) {
        printf("Wrong argument format\n");
        printf("Usage: ./mydisambig -text testdata/$$i.txt -map ZhuYin-Big5.map -lm bigram.lm -order 2 > result2/$$i.txt\n");
        exit(1);
    }

    char *ptr;
    const int ngram_order = strtol(argv[8], &ptr, 10);
    int i, t;

    Vocab voc, zhuyin, big5;
    VocabIndex voc_index, zhuyin_index, big5_index;

    /**
     * Read language model
     */
    Ngram lm(voc, ngram_order);
    {
        const char *lm_filename = argv[6];
        File lm_file(lm_filename, "r");
        lm.read(lm_file);
        lm_file.close();
    }

    /**
     * Read map
     */
    VocabMap map(zhuyin, big5);	
    {
        const char *map_filename = argv[4];
        File map_file(map_filename, "r");
        map.read(map_file);
        map_file.close();
    }
    
    /**
     * Read test data
     */
    const char *test_filename = argv[2];	
    File test_file(test_filename, "r");
    char *line;

    while (line = test_file.getline()) {
        VocabString words[maxWordsPerLine];
        unsigned int words_length;
        
        words_length = Vocab::parseWords(line, &(words[1]), maxWordsPerLine);
        words[0] = Vocab_SentStart; // Vocab_SentStart = "<s>"
        words[words_length+1] = Vocab_SentEnd; // Vocab_SentEnd = "</s>"
        words_length += 2;
        // words = ["<s>", "w_1", "w_2", ..., "w_n", "</s>"]

        /**
         * Start running Viterbi algorithm
         * 1) Initialization
         * 2) Recursion
         * 3) Termination
         * 4) Path backtracking
         */

        /**
         * 1) Initialization
         */
        LogP delta[words_length][MAX_CANDIDATE];
        VocabIndex delta_index[words_length][MAX_CANDIDATE]; // Store big5 index in each entry
        int psi[words_length][MAX_CANDIDATE]; // For path backtracking
        int num_candidate[words_length]; // Number of candidate at each time step

        // All sentence start from "<s>" with probability of 1
        delta[0][0] = LogP_One;
        delta_index[0][0] = big5.getIndex(Vocab_SentStart);
        num_candidate[0] = 1;

        /**
         * 2) Recursion
         */
        for (t = 1; t < words_length; t++) {
            VocabMapIter map_iter(map, zhuyin.getIndex(words[t])); // VocabMapIter(VocabMap &vmap, VocabIndex w);			
            VocabString w1, w2;
            LogP prob, max_prob;
            Prob p; // Useless
            int best_w1, candidate_cnt = 0;

            while (map_iter.next(big5_index, p)) { // VocabMapIter.next(VocabIndex &w, Prob &prob);
                w2 = big5.getWord(big5_index); // w2 = candidate word
                if (voc.getIndex(w2) == Vocab_None) {
                    w2 = Vocab_Unknown;
                }

                // Iterate over every candidates at last time step
                max_prob = LogP_Zero;
                for (i = 0; i < num_candidate[t-1]; i++) {
                    w1 = big5.getWord(delta_index[t-1][i]);
                    prob = getBigramProb(voc, lm, w1, w2);
                    // If prob too small, backoff to unigram
                    if (prob == LogP_Zero) { // TODO: check if small enough
                        prob = getUigramProb(voc, lm, w2);
                    }
                    
                    // Get total prob
                    prob += delta[t-1][i];

                    if (prob > max_prob) {
                        max_prob = prob;
                        best_w1 = i;
                    }
                }

                delta[t][candidate_cnt] = max_prob;
                delta_index[t][candidate_cnt] = big5.getIndex(w2);
                psi[t][candidate_cnt] = best_w1;

                candidate_cnt++;
            }
            
            num_candidate[t] = candidate_cnt;
        }

        /**
         * 3) Termination
         */
        VocabString output_words[words_length];
        LogP max_prob = LogP_Zero;
        int bt_index; // Backtrack index

        for (i = 0; i < num_candidate[words_length-1]; i++) {
            if (delta[words_length-1][i] > max_prob) {
                max_prob = delta[words_length-1][i];
                output_words[words_length-1] = big5.getWord(delta_index[words_length-1][i]);
                bt_index = i;
            }
        }

        /**
         * 4) Path backtracking
         */
        for (t = words_length - 2; t >= 0; t--) {
            bt_index = psi[t+1][bt_index];
            output_words[t] = big5.getWord(delta_index[t][bt_index]);
        }

        /**
         * Check if words change from big5 to <unk>,
         * return them back to original words
         */
        for (t = 0; t < words_length; t++) {
            if (strcmp(words[t], Vocab_Unknown) && !strcmp(output_words[t], Vocab_Unknown)) {
                output_words[t] = words[t];
            }
        }

        /**
         * Output final sentence
         */
        for (t = 0; t < words_length; t++) {
            printf("%s", output_words[t]);
            if (t == words_length - 1) {
                printf("\n");
            } else {
                printf(" ");
            }
        }
    }

    test_file.close();
    
    return 0;
}