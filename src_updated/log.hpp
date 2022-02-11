#include <mutex>
#include "thread_pool.hpp"
#include <windows.h>

//#pragma comment(lib,"winmm")
#define TIMER_UDELAY		1
#define TIMER_RESOLUTION	0
#define TIMER_ITERATION		50000

mutex cout_guard;

void cout_log(){
    lock_guard<mutex> lk(cout_guard);
    cout << tim() << " " << thread_pool.n_idle() << endl;
}

void CALLBACK timerProc(UINT uTimerID,UINT uMsg,DWORD_PTR dwUser,DWORD_PTR dw1,DWORD_PTR dw2) {
	cout_log();
}

void set_timer(){
    int ti = 0;
    MMRESULT timerID = timeSetEvent(
		TIMER_UDELAY,
		TIMER_RESOLUTION,
		timerProc,
		(DWORD_PTR)&ti,
		TIME_PERIODIC | TIME_CALLBACK_FUNCTION
	);
}
