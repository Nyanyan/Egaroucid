// INCLUDE MUST BE MODIFIED
#include <Python.h>
// example
//#include "C:/Users/username/AppData/Local/Programs/Python/Python39/include/Python.h"

#include <iostream>
#include "ai.hpp"

using namespace std;

static PyObject* egaroucid5_init(PyObject* self, PyObject* args){
    char* eval_file;
    char* book_file;
    if (!PyArg_ParseTuple(args, "ss", &eval_file, &book_file)){
        return NULL;
    }
    string book_file_str = string(book_file);
    bit_init();
    flip_init();
    board_init();
    bool res;
    if (!evaluate_init(eval_file))
        res = false;
    if (res)
        res = book_init(book_file_str);
    return res ? Py_True : Py_False;
}

static PyObject* egaroucid5_ai(PyObject* self, PyObject* args){
    char* board_c_str;
    int level;
    if (!PyArg_ParseTuple(args, "si", &board_c_str, &level)){
        return NULL;
    }
    string board_str_with_space = string(board_c_str);
    string board_str;
    for (int i = 0; i < board_str_with_space.size(); ++i){
        if (board_str_with_space[i] != ' ')
            board_str += board_str_with_space[i];
    }
    if (board_str.size() != HW2 + 1)
        return Py_None;
    int bd_arr[HW2], player;
	Board bd;
    bool flag = true;
    for (int i = 0; i < HW2; ++i) {
        if (board_str[i] == '0' || board_str[i] == 'B' || board_str[i] == 'b' || board_str[i] == 'X' || board_str[i] == 'x' || board_str[i] == '*')
            bd_arr[i] = BLACK;
        else if (board_str[i] == '1' || board_str[i] == 'W' || board_str[i] == 'w' || board_str[i] == 'O' || board_str[i] == 'o')
            bd_arr[i] = WHITE;
        else if (board_str[i] == '.' || board_str[i] == '-')
            bd_arr[i] = VACANT;
        else {
            flag = false;
            break;
        }
    }
    if (board_str[HW2] == '0' || board_str[HW2] == 'B' || board_str[HW2] == 'b' || board_str[HW2] == 'X' || board_str[HW2] == 'x' || board_str[HW2] == '*')
        player = BLACK;
    else if (board_str[HW2] == '1' || board_str[HW2] == 'W' || board_str[HW2] == 'w' || board_str[HW2] == 'O' || board_str[HW2] == 'o')
        player = WHITE;
    else
        flag = false;
	if (flag) {
		bd.translate_from_arr(bd_arr, player);
        if (bd.get_legal()){
            Search_result res = ai(bd, level, true, 0);
            return Py_BuildValue("is", res.value, idx_to_coord(res.policy));
        }
	}
    return Py_None;
}

static PyMethodDef egaroucid5Methods[] = {
    {"init", egaroucid5_init, METH_VARARGS, "Egaroucid 5 initialize"},
    {"ai", egaroucid5_ai, METH_VARARGS, "Egaroucid 5 calculate best move and score"},
    {NULL}
};

// myModule definition struct
static struct PyModuleDef egaroucid5Module = {
    PyModuleDef_HEAD_INIT,
    "egaroucid5",
    "Egaroucid 5 AI",
    -1,
    egaroucid5Methods
};

// Initializes myModule
PyMODINIT_FUNC PyInit_egaroucid5(void){
    return PyModule_Create(&egaroucid5Module);
}