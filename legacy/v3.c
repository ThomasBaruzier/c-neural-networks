#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define DEBUG
#define MB 1024*1024

#define sigmoid(x) (1 / (1 + exp(-x)))
#define dSigmoid(x) (x * (1 - x))
#define relu(x) ((x > 0) ? x : 0)
#define dRelu(x) ((x > 0) ? 1 : 0)
#define randomValue() ((double)rand())/((double)RAND_MAX)

struct HyperParams {
  size_t numInputs;
  size_t numOutputs;
  size_t numSets;
  size_t numEpochs;
  size_t numLayers;
  size_t *layerSizes;
  float learningRate;
};

struct LoadParams {
  size_t arraySize;
  size_t set;
  size_t value;
  size_t line;
  size_t pos;
  char buffer[5*MB];
  char step;
};

struct Neurons {
  float (*dataset)[1][1];
  float ***weights;
  float **biases;
  float **values;
};

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

size_t countNeurons(char *buffer, size_t pos) {
  int neurons=1;
  for (; buffer[pos] != ' ' && buffer[pos] != '\0'; pos++) {
    if (buffer[pos] == ',') neurons++;
  }
  return neurons;
}

int allocArrays(struct Neurons *n, struct HyperParams *hp, struct LoadParams *lp) {
  size_t toAlloc;
  if (lp->set == lp->arraySize) lp->step = '3';

  switch (lp->step) {
    case '1':
      hp->numInputs = countNeurons(lp->buffer, 0);
      toAlloc = (sizeof(float) * 2 * lp->arraySize * hp->numInputs);
//      printf("Step 1 : Allocating %zu bytes\n", toAlloc);
      n->dataset = malloc(toAlloc);
      break;
    case '2':
      hp->numOutputs = countNeurons(lp->buffer, lp->pos);
      toAlloc = sizeof(float) * 4 * lp->arraySize * hp->numInputs;
//      printf("Step 2 : Allocating %zu bytes\n", toAlloc);
      n->dataset = realloc(n->dataset, toAlloc);
      break;
    case '3':
      lp->arraySize *= 2;
      toAlloc = sizeof(float) * 4 * lp->arraySize * hp->numInputs;
//      printf("Step 3 : Reallocating %zu bytes\n", toAlloc);
      n->dataset = realloc(n->dataset, toAlloc);
  }

  if (n->dataset == NULL) {
    printf("Error while allocating memory");
    exit(1);
  }

  lp->step = '0';
  return 0;
}

int loadData(FILE *fp, struct Neurons *n, struct HyperParams *hp) {
  struct LoadParams lp;
  lp.set = 0; lp.value = 0; lp.line = 0; lp.pos = 0;
  lp.arraySize = 8; lp.step = '1';
  char tmp[8]; size_t i;

  while (fread(lp.buffer, 1, 5*MB, fp) > 0) {
    for (lp.pos = 0; lp.line < 2 && lp.buffer[lp.pos] != '\0'; lp.pos++) {
      allocArrays(n, hp, &lp);

      for (i=0; i < 8; i++) tmp[i] = '\0'; i=0;
      while ((lp.buffer[lp.pos] >= '0' && lp.buffer[lp.pos] <= '9') || lp.buffer[lp.pos] == '.') {
        tmp[i] = lp.buffer[lp.pos]; lp.pos++; i++;
      }

//      printf("n->dataset[%zu][%zu][%zu] = %f\n", lp.line, lp.set, lp.value, atof(tmp));
      n->dataset[lp.line][lp.set][lp.value] = atof(tmp);
      if (lp.buffer[lp.pos] == ',') { lp.value++; }
      else if (lp.buffer[lp.pos] == ' ') { lp.set++; lp.value=0; }
      else if (lp.buffer[lp.pos] == '\n' || lp.buffer[lp.pos] == '\0') {
        if (lp.line == 0) { hp->numInputs = lp.value+1; lp.step = '2'; }
        if (lp.line == 1) { hp->numOutputs = lp.value+1; hp->numSets = lp.set+1; }
        lp.line++; lp.set = 0; lp.value = 0;
      }
    }
  }

//  printf("Input layer size : %zu\n", hp->numInputs);
//  printf("Output layer size : %zu\n", hp->numOutputs);
//  printf("Number of training sets : %zu\n\n", hp->numSets);
  return 0;
}

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

    struct HyperParams hp;
    struct Neurons n;
    loadData(fp, &n, &hp);
    // future code

  } else {
    printf("Running %s\n", path);
    // future code

  }

  return 0;
}
