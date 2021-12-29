/** search **/


/* cutting */

// multi prob cut
#define USE_MID_MPC true
#define USE_END_MPC true

// stability cut
#define USE_MID_SC  false
#define USE_END_SC  false

// transpose table cut
#define USE_MID_TC true
#define USE_END_TC true


/* ordering */

// parity ordering
#define USE_END_PO true


/* smoothing */
#define USE_MID_SMOOTH true



/* multi threading*/

// use multi thread ** UNDER CONSTRUCTION **
#define USE_MULTI_THREAD false

// YBWC early cut **If you want to set these true, you must set USE_MULTI_THREAD true**
#define USE_YBWC_MID_EARLY_CUT true
#define USE_YBWC_END_EARLY_CUT true







/** book **/
#define USE_BOOK true








/** other **/

// MPC calculation mode
#define MPC_MODE true

// creatring evaluation data mode
#define EVAL_MODE false

// book mode
#define BOOK_MODE false