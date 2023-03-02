#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
int countNeurons(char *set);
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
    printf("Training using %s\n", path);

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

int countNeurons(char *set) {
  int i=0, neurons=1;
  while(set[i] != '\0') {
    if (set[i] == ',') neurons++;
    i++;
  }
  return neurons;
}

// TO DO
// reduce alloc calls
// handle fread chunks
// resize inputs/outputs dynamically
int loadData(FILE *fp, struct Dataset *ds, struct HyperParams *hp) {
  char buffer[1024*1024], tmp[8];
  int numNeurons, flag=0;
  int i=0, j=0, k=0, l=0, x=0;

  // allocate ds->inputs
  fscanf(fp, "%s", buffer);
  numNeurons = countNeurons(buffer);
  ds->inputs = (float **) malloc(1000 * sizeof(float *));
  for (int i = 0; i < 1000; i++)
    ds->inputs[i] = (float *) malloc(numNeurons * sizeof(float));


  while (fread(buffer, 1, 1024*1024, fp) > 0) {
    for (int i=0; j < 2 && buffer[i] != '\0'; i++) {

      // allocate ds->outputs
      if (flag) {
        flag = 0;
        fscanf(fp, "%s", buffer);
        numNeurons = countNeurons(buffer);
        ds->outputs = (float **) malloc(1000 * sizeof(float *));
        for (int i = 0; i < 1000; i++)
          ds->outputs[i] = (float *) malloc(numNeurons * sizeof(float));
      }

      // parsing doubles
      for (x=0; x < 8; x++) tmp[x] = '\0'; x=0;
      while ((buffer[i] >= '0' && buffer[i] <= '9') || buffer[i] == '.') {
        tmp[x] = buffer[i]; i++; x++;
      }

      // assigning values
      if (j == 0) ds->inputs[k][l] = atof(tmp);
      else ds->outputs[k][l] = atof(tmp);
      if (j == 0) printf("inputs[%d][%d]=%f\n", k, l, ds->inputs[k][l]);
      else printf("outputs[%d][%d]=%f\n", k, l, ds->outputs[k][l]);;

      // handle separators and counters
      if (buffer[i] == ',') { l++; }
      else if (buffer[i] == ' ') { k++; l=0; }
      else if (buffer[i] == '\n' || buffer[i] == '\0') {
        if (j == 0) { hp->numInputs = l+1; flag=1; }
        if (j == 1) { hp->numOutputs = l+1; hp->numSets = k+1; }
        j++; k=0; l=0;
      }
    }
  }

  return 0;
}

