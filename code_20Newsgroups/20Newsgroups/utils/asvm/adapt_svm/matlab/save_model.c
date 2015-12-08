#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mex.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

void mexFunction( int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[] )
{
	struct svm_model *model;
    const char *error_msg;
    char filename[256]; 
	if(nrhs != 2)
	{
		mexPrintf("Error: incorrect input arguments!\n");
		return;
	}

    mxGetString(prhs[0], filename, mxGetN(prhs[0])+1);
	model = matlab_matrix_to_model(prhs[1], &error_msg);
	if (model == NULL)
	{
		mexPrintf("Error: can't read model: %s\n", error_msg);
        return;
	}
    svm_save_model(filename, model);
    svm_free_and_destroy_model(&model);
}
