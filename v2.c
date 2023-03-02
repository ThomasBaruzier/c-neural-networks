#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG

#define sigmoid(x) (1 / (1 + exp(-x)))
#define dSigmoid(x) (x * (1 - x))
#define relu(x) ((x > 0) ? x : 0)
#define dRelu(x) ((x > 0) ? 1 : 0)
#define randomValue() ((double)rand())/((double)RAND_MAX)

struct HyperParams {
  int numInputs;
  int numOutputs;
  int numSets;
  int numEpochs;
  int numLayers;
  int *layerSizes;
  float learningRate;
};

struct Dataset {
  float **inputs;
  float **outputs;
};

struct Neurons {
  float ***weights;
  float **biases;
  float **values;
};

int detectMode(char *path);
int countNeurons(char *set, int pos);
int loadData(FILE *fp, struct Dataset *ds, struct HyperParams *hp);

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage : %s <dataset.ds>|<model.ml>\n", argv[0]);
    return 1;
  }

  char *path = argv[1];
  FILE *fp = fopen(path, "r");
  if (fp == NULL) {
    printf("ERROR : File doesn't exist\n");
    exit(1);
  }

  if (detectMode(path)) {
    printf("\nTraining using %s\n\n", path);

    struct Dataset ds;
    struct HyperParams hp;
    struct Neurons n;
    loadData(fp, &ds, &hp);
    // future code

  } else {
    printf("Running %s\n", path);
    // future code

  }

  return 0;
}

int detectMode(char *path) {
  int i;
  for (i=0; path[i+2] != '\0'; i++);
  if (path[i] == 'd' && path[i+1] == 's') return 1;
  if (path[i] == 'm' && path[i+1] == 'l') return 0;
  else {
    printf("ERROR : Unrecognized file extension\n");
    exit(1);
  }
}

int countNeurons(char *set, int pos) {
  int neurons=1;
  for (; set[pos] != ' ' && set[pos] != '\0'; pos++) {
    if (set[pos] == ',') neurons++;
  }
  return neurons;
}

// TO DO
// Fix allocations issues (random garbage values from time to time)
int loadData(FILE *fp, struct Dataset *ds, struct HyperParams *hp) {
  char buffer[1024*1024], tmp[8];
  float *inputData = NULL, *outputData = NULL;
  int numNeuronsIn, numNeuronsOut, arraySize = 1024;
  int i, line=0, set=0, value=0, step=0;
  ds->inputs = (float **) malloc(arraySize * sizeof(float *));
  ds->outputs = (float **) malloc(arraySize * sizeof(float *));

  while (fread(buffer, 1, 1024*1024, fp) > 0) {
    for (int pos=0; line < 2 && buffer[pos] != '\0'; pos++) {

      // dynamic allocation based on set sizes
      if (step == 0) {
        numNeuronsIn = countNeurons(buffer, 0);
        inputData = (float *) realloc(inputData, arraySize * numNeuronsIn * sizeof(float));
        step = -1;
      } else if (step == 1) {
        arraySize = 1024;
        numNeuronsOut = countNeurons(buffer, pos);
        outputData = (float *) realloc(outputData, arraySize * numNeuronsOut * sizeof(float));
        step = -1;
      }

      // parsing doubles
      for (i=0; i < 8; i++) tmp[i] = '\0'; i=0;
      while ((buffer[pos] >= '0' && buffer[pos] <= '9') || buffer[pos] == '.') {
        tmp[i] = buffer[pos]; pos++; i++;
      }

      // assigning values
      if (line == 0)  {
        // reallocate inputs
        if (set == arraySize) {
          arraySize *= 2;
          ds->inputs = (float **) realloc(ds->inputs, arraySize * sizeof(float *));
          inputData = (float *) realloc(inputData, arraySize * numNeuronsIn * sizeof(float));
        }

        ds->inputs[set] = inputData + set * numNeuronsIn;
        ds->inputs[set][value] = atof(tmp);
        #ifdef DEBUG
          printf("inputs[%d][%d]=%f\n", set, value, ds->inputs[set][value]);
        #endif
      } else if (line == 1) {
        // reallocate outputs
        if (set == arraySize) {
          arraySize *= 2;
          ds->outputs = (float **) realloc(ds->outputs, arraySize * sizeof(float *));
          outputData = (float *) realloc(outputData, arraySize * numNeuronsOut * sizeof(float));
        }

        ds->outputs[set] = outputData + set * numNeuronsOut;
        ds->outputs[set][value] = atof(tmp);
        #ifdef DEBUG
          printf("outputs[%d][%d]=%f\n", set, value, ds->inputs[set][value]);
        #endif
      }

      // handle separators and counters
      if (buffer[pos] == ',') { value++; }
      else if (buffer[pos] == ' ') { set++; value=0; }
      else if (buffer[pos] == '\n' || buffer[pos] == '\0') {
        if (line == 0) { hp->numInputs = value+1; step = 1; }
        if (line == 1) { hp->numOutputs = value+1; hp->numSets = set+1; }
        line++; set = 0; value = 0;
      }
    }
  }

  #ifdef DEBUG
    printf("\nnumInputs = %d\n", hp->numInputs);
    printf("numOutputs = %d\n", hp->numOutputs);
    printf("numSets = %d\n\n", hp->numSets);
  #endif

  return 0;
}

//printf("char : %c | pos : %d | number : %f | line : %d | set : %d | value : %d\n", buffer[pos], pos, atof(tmp), line, set, value);
