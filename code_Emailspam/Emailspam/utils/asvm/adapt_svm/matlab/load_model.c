#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../svm.h"

#include "mex.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}


struct svm_model *model;

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;

    int nr_feat = 0;
    int i;
	if(nrhs == 1)
	{
		char filename[256];
		mxGetString(prhs[0], filename, mxGetN(prhs[0]) + 1);
		if(filename == NULL)
		{
			mexPrintf("Error: filename is NULL\n");
			return;
		}
        model = svm_load_model(filename);
        if (model->param.kernel_type == PRECOMPUTED)
            nr_feat = 1;
        else
        {
            for(i = 0; ;i++)
            {
                if(model->SV[0][i].index != -1)
                    nr_feat++;
                else
                    break;
            }
        }
        error_msg = model_to_matlab_structure(plhs, nr_feat, model);
        if(error_msg)
            mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
        svm_free_and_destroy_model(&model);
	}
	else
	{
		fake_answer(plhs);
	}
}
