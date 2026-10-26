#include "ecg_model.h"

#include <string.h>

#include "ecg_stage.h"
#include "ecga.h"
#include "ecga_data.h"
#include "ecgb.h"
#include "ecgb_data.h"
#include "model_meta.h"
#include "platform.h"

static ai_handle net_a = AI_HANDLE_NULL, net_b = AI_HANDLE_NULL;
static ai_buffer *a_in, *a_out, *b_in, *b_out;
static const int b_order[AI_ECGB_IN_NUM] = ECGB_C_INPUT_ORDER;
static float stage_rr[STAGE_N_LAGS];

int ecg_model_init(void *act_a, void *act_b)
{
    const ai_handle acts_a[] = {act_a};
    const ai_handle acts_b[] = {act_b};
    if (ai_ecga_create_and_init(&net_a, acts_a, NULL).type != AI_ERROR_NONE)
        return -1;
    if (ai_ecgb_create_and_init(&net_b, acts_b, NULL).type != AI_ERROR_NONE)
        return -2;
    a_in = ai_ecga_inputs_get(net_a, NULL);
    a_out = ai_ecga_outputs_get(net_a, NULL);
    b_in = ai_ecgb_inputs_get(net_b, NULL);
    b_out = ai_ecgb_outputs_get(net_b, NULL);
    if (!a_in || !a_out || !b_in || !b_out)
        return -3;
    return 0;
}

/* Net B input slot that takes the tensor meaning `which` (ECGB_IN_*). */
static float *b_input(int which)
{
    for (int i = 0; i < AI_ECGB_IN_NUM; i++)
        if (b_order[i] == which)
            return (float *)b_in[i].data;
    return NULL;
}

static void copy_out(int which_b, int which_a)
{
    memcpy(b_input(which_b), a_out[which_a].data, AI_BUFFER_SIZE(&a_out[which_a]) * sizeof(float));
}

int ecg_model_run(const float *x, ecg_model_out_t *out)
{
    uint32_t t0 = plat_micros();
    memcpy(a_in[0].data, x, MODEL_SAMPLES * MODEL_LEADS * sizeof(float));
    if (ai_ecga_run(net_a, a_in, a_out) != 1)
        return -1;
    uint32_t t1 = plat_micros();

    /* net A's tensors go straight into net B's input slots (separate activation buffers) */
    float *qrs = b_input(ECGB_IN_QRS);
    ecg_stage_run(x, (const float *)a_out[ECGA_OUT_P_POS].data, b_input(ECGB_IN_RESIDUAL), qrs,
                  stage_rr);
    memcpy(b_input(ECGB_IN_RR), stage_rr, sizeof(stage_rr));
    copy_out(ECGB_IN_POS_LOGIT, ECGA_OUT_POS_LOGIT);
    copy_out(ECGB_IN_V1, ECGA_OUT_V1);
    copy_out(ECGB_IN_V2, ECGA_OUT_V2);
    copy_out(ECGB_IN_V3, ECGA_OUT_V3);
    copy_out(ECGB_IN_VB, ECGA_OUT_VB);
    copy_out(ECGB_IN_VS, ECGA_OUT_VS);
    copy_out(ECGB_IN_CHAN, ECGA_OUT_CHAN);
    uint32_t t2 = plat_micros();

    if (ai_ecgb_run(net_b, b_in, b_out) != 1)
        return -2;
    uint32_t t3 = plat_micros();
    out->rhythm = (const float *)b_out[0].data;     /* C output order = TFLite order */
    out->beat = (const float *)b_out[1].data;
    out->channel = (const float *)b_out[2].data;
    out->us_a = t1 - t0;
    out->us_stage = t2 - t1;
    out->us_b = t3 - t2;
    return 0;
}
