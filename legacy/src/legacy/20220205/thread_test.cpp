#include <iostream>
#include <mutex>
#include <chrono>
#include "CTPL/ctpl_stl.h" // from https://github.com/vit-vit/CTPL

using namespace std;

ctpl::thread_pool thread_pool(8);

#define n 10000000

class global_num{
    private:
        int num;
        mutex mtx;
    public:
        void write(int x){
            lock_guard<mutex> lock(mtx);
            num = x;
        }

        int read(){
            lock_guard<mutex> lock(mtx);
            return num;
        }

};

class local_num{
    private:
        int num;
    public:
        void write(int x){
            num = x;
        }

        int read(){
            return num;
        }

};

global_num g;

inline long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void do_task(int id){
    int a;
    for (int i = 0; i < n; ++i){
        //a = g.read();
        g.write(i);
    }
}

void do_task2(local_num l){
    for (int i = 0; i < n; ++i){
        l.write(i);
    }
}

void print_addr(local_num l){
    cout << &l << endl;
}

int main(){
    local_num lo;
    cout << &lo << endl;
    print_addr(lo);
    return 0;


    long long strt = tim();
    for (int i = 0; i < 8; ++i){
        do_task(i);
    }
    cout << tim() - strt << endl;

    strt = tim();
    future<void> tasks[8];
    for (int i = 0; i < 8; ++i){
        tasks[i] = thread_pool.push(do_task);
    }
    for (int i = 0; i < 8; ++i){
        tasks[i].get();
    }
    cout << tim() - strt << endl;
    
    strt = tim();
    local_num l;
    for (int i = 0; i < 8; ++i){
        tasks[i] = thread_pool.push(bind(&do_task2, l));
    }
    for (int i = 0; i < 8; ++i){
        tasks[i].get();
    }
    cout << tim() - strt << endl;

    return 0;
}
