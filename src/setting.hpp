/** search **/


/* cutting */

// multi prob cut
#define USE_MID_MPC false
#define USE_END_MPC false

// stability cut
#define USE_MID_SC  false
#define USE_END_SC  true

// transpose table cut
#define USE_MID_TC true
#define USE_END_TC true

// YBWC early cut **If you want to set these true, you must set USE_MULTI_THREAD true**
#define USE_YBWC_MID_EARLY_CUT true
#define USE_YBWC_END_EARLY_CUT true


/* ordering */

// parity ordering
#define USE_END_PO true



/* multi threading*/

// use multi thread ** UNDER CONSTRUCTION **
#define USE_MULTI_THREAD false



/** other **/

// MPC calculation mode
#define MPC_MODE false

// creatring evaluation data mode
#define EVAL_MODE false
