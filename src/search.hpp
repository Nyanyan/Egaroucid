#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#define search_epsilon 1
constexpr int cache_hit = 100;
constexpr int cache_both = 100;
constexpr int parity_vacant_bonus = 10;
constexpr int canput_bonus = 1;
//constexpr int mtd_threshold = 0;
constexpr int mtd_end_threshold = 5;

#define mpc_min_depth 3
#define mpc_max_depth 10
#define mpc_min_depth_final 9
#define mpc_max_depth_final 28

#define simple_mid_threshold 3
#define simple_end_threshold 7

#define po_max_depth 8

#define extra_stability_threshold 58

#define ybwc_mid_first_num 1
#define ybwc_end_first_num 2
#define multi_thread_depth 1

const int cell_weight[hw2] = {
    10, 3, 9, 7, 7, 9, 3, 10, 
    3, 2, 4, 5, 5, 4, 2, 3, 
    9, 4, 8, 6, 6, 8, 4, 9, 
    7, 5, 6, 0, 0, 6, 5, 7, 
    7, 5, 6, 0, 0, 6, 5, 7, 
    9, 4, 8, 6, 6, 8, 4, 9, 
    3, 2, 4, 5, 5, 4, 2, 3, 
    10, 3, 9, 7, 7, 9, 3, 10
};

const int mpcd[30] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9};
#if USE_MID_SMOOTH
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {0.651750535912282, 0.5589463177207203, 0.640370223128342, 0.8053205039351448, 1.1272003571509772, 1.2471361386641218, 1.5014470377269764, 1.4579291148612596},
        {1.3072598756651512, 1.4559478605574687, 1.3071416811207823, 1.381596093197997, 1.3494063093918456, 1.5312487450252241, 1.5830225092842687, 1.407233186331862},
        {2.3449912850550483, 2.283852401390813, 1.849754677182859, 2.0169969470863744, 2.134783911026133, 2.5809729663295986, 2.432165407908881, 2.474831991960212},   
        {2.3687443113183617, 2.498461794770302, 2.096012243936727, 2.659442636180584, 2.4218232738902787, 2.340485237821291, 3.6247406117126673, 2.851725496687525},   
        {2.6440501366009292, 2.449321039332943, 2.2347744805572227, 3.3054581588407292, 3.2203082428956438, 2.859460810892896, 4.08929692632364, 3.2352399022307727},  
        {3.4681278358511274, 3.852854832215551, 2.6550415871672244, 3.902620481560266, 3.146037748791259, 3.007136354997551, 3.4065348686570585, 2.9839172290374076},  
        {3.4289534362025584, 3.1712520967034683, 2.878642972974186, 3.786502135902851, 4.192691311811194, 3.842619610144816, 4.060260371665133, 4.570116509705272},    
        {4.365512923040372, 2.742707072390722, 3.023276815427708, 5.013507924396747, 4.33907153980455, 4.411097816454943, 4.431737386679314, 5.490577141470407},       
        {4.464384190588743, 3.877284272958759, 3.7200329527241864, 6.345778719985846, 4.717386071044386, 5.043137232643338, 5.880114299840816, 6.6898304762238245},    
        {3.8321384493098547, 3.0563371631074174, 3.487588736333235, 4.5285712492663395, 2.747905989448947, 1.5944663165603936, 4.7784690026199454, 2.321043221023311}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        5.300496203921781, 5.086744392045486, 5.653462288187769, 5.493576365741511, 5.392302288637843, 5.840969628500125, 5.5492649342214655, 5.30440023973874, 5.288926106221303, 5.237455881235005, 4.89209516955069, 4.901166825767374, 4.73048964991016, 5.115506839212132, 5.097766547352733, 4.910173132200829, 5.147450187378907, 5.231400707385932, 5.355386151684506, 5.670606356378616
    };
#else
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
    };
#endif
unsigned long long can_be_flipped[hw2];

unsigned long long searched_nodes;
vector<int> vacant_lst;

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

inline void mpc_init(){
    int i, j;
    for (i = 0; i < n_phases; ++i){
        for (j = 0; j < mpc_max_depth - mpc_min_depth + 1; ++j)
            mpcsd[i][j] /= step;
    }
    for (i = 0; i < mpc_max_depth_final - mpc_min_depth_final + 1; ++i)
        mpcsd_final[i] /= step;
}

inline void search_init(){
    //mpc_init();
    int i;
    for (int cell = 0; cell < hw2; ++cell){
        can_be_flipped[cell] = 0b1111111110000001100000011000000110000001100000011000000111111111;
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][0]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][0]][i];
        }
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][1]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][1]][i];
        }
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][2]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][2]][i];
        }
        if (place_included[cell][3] != -1){
            for (i = 0; i < hw; ++i){
                if (global_place[place_included[cell][3]][i] != -1)
                    can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][3]][i];
            }
        }
    }
    cerr << "search initialized" << endl;
}

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline void move_ordering(board *b){
    int l, u;
    transpose_table.get_prev(b, b->hash() & search_hash_mask, &l, &u);
    if (u != inf && l != -inf)
        b->v = (u + l) / 2 + cache_hit + cache_both;
    else if (u != inf)
        b->v += u + cache_hit;
    else if (l != -inf)
        b->v += l + cache_hit;
    else
        b->v = -mid_evaluate(b);
}

inline void move_ordering_eval(board *b){
    b->v = -mid_evaluate(b);
}

inline void calc_extra_stability(board *b, int p, unsigned long long extra_stability, int *pres, int *ores){
    *pres = 0;
    *ores = 0;
    int y, x;
    extra_stability >>= hw;
    for (y = 1; y < hw_m1; ++y){
        extra_stability >>= 1;
        for (x = 1; x < hw_m1; ++x){
            if ((extra_stability & 1) == 0){
                if (pop_digit[b->b[y]][x] == p)
                    ++*pres;
                else if (pop_digit[b->b[y]][x] == 1 - p)
                    ++*ores;
            }
            extra_stability >>= 1;
        }
        extra_stability >>= 1;
    }
}

inline unsigned long long calc_extra_stability_ull(board *b){
    unsigned long long extra_stability = 0b1111111110000001100000011000000110000001100000011000000111111111;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == vacant)
            extra_stability |= can_be_flipped[cell];
    }
    return extra_stability;
}

inline bool stability_cut(board *b, int *alpha, int *beta){
    if (b->n >= extra_stability_threshold){
        int ps, os;
        calc_extra_stability(b, b->p, calc_extra_stability_ull(b), &ps, &os);
        *alpha = max(*alpha, (2 * (calc_stability(b, b->p) + ps) - hw2));
        *beta = min(*beta, (hw2 - 2 * (calc_stability(b, 1 - b->p) + os)));
    } else{
        *alpha = max(*alpha, (2 * calc_stability(b, b->p) - hw2));
        *beta = min(*beta, (hw2 - 2 * calc_stability(b, 1 - b->p)));
    }
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    int res = 0;
    for (const int &cell: vacant_lst)
        res += b->legal(cell);
    return res;
}
