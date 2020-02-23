
#include <cstdio>
#include <cmath>
#include <iostream>
#include <boost/timer/timer.hpp>
double black(double t, double* d_t, double T, double* d_T, double r, double* d_r, double S, double* d_S, double K, double* d_K, double vol, double* d_vol)
{
    double __statement_0 = ((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*(((std::log(((S)/(K))))+(((((r)+(((std::pow(vol, 2))/(2)))))*(((T)-(t))))))));
    double __temp_0 = (((((((-(((vol)*(((0.5)*(((std::pow(((T)-(t)), -0.5))*(-1)))))))))/(std::pow(((vol)*(std::pow(((T)-(t)), 0.5))), 2))))*(((std::log(((S)/(K))))+(((((r)+(((std::pow(vol, 2))/(2)))))*(((T)-(t)))))))))+(((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*(((((r)+(((std::pow(vol, 2))/(2)))))*(-1))))));
    double __diff___statement_0_t = __temp_0;
    double __temp_7 = (((((((-(((vol)*(((0.5)*(std::pow(((T)-(t)), -0.5))))))))/(std::pow(((vol)*(std::pow(((T)-(t)), 0.5))), 2))))*(((std::log(((S)/(K))))+(((((r)+(((std::pow(vol, 2))/(2)))))*(((T)-(t)))))))))+(((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*(((r)+(((std::pow(vol, 2))/(2))))))));
    double __diff___statement_0_T = __temp_7;
    double __temp_14 = ((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*(((T)-(t))));
    double __diff___statement_0_r = __temp_14;
    double __temp_21 = ((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*(((((K)/(std::pow(K, 2))))/(((S)/(K))))));
    double __diff___statement_0_S = __temp_21;
    double __temp_28 = ((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*((((((-(S)))/(std::pow(K, 2))))/(((S)/(K))))));
    double __diff___statement_0_K = __temp_28;
    double __temp_35 = (((((((-(std::pow(((T)-(t)), 0.5))))/(std::pow(((vol)*(std::pow(((T)-(t)), 0.5))), 2))))*(((std::log(((S)/(K))))+(((((r)+(((std::pow(vol, 2))/(2)))))*(((T)-(t)))))))))+(((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*(((((((((2)*(vol)))*(2)))/(4)))*(((T)-(t))))))));
    double __diff___statement_0_vol = __temp_35;



    double __statement_1 = ((__statement_0)-(((vol)*(((T)-(t))))));
    double __temp_36 = (-(((vol)*(-1))));
    double __temp_42 = __diff___statement_0_t;
    double __diff___statement_1_t = __temp_36 + __temp_42;
    double __temp_44 = (-(vol));
    double __temp_49 = __diff___statement_0_T;
    double __diff___statement_1_T = __temp_44 + __temp_49;
    double __temp_56 = __diff___statement_0_r;
    double __diff___statement_1_r = __temp_56;
    double __temp_63 = __diff___statement_0_S;
    double __diff___statement_1_S = __temp_63;
    double __temp_70 = __diff___statement_0_K;
    double __diff___statement_1_K = __temp_70;
    double __temp_76 = (-(((T)-(t))));
    double __temp_77 = __diff___statement_0_vol;
    double __diff___statement_1_vol = __temp_76 + __temp_77;



    double __statement_2 = ((K)*(std::exp((((-(r)))*(((T)-(t)))))));
    double __temp_78 = ((K)*(((std::exp((((-(r)))*(((T)-(t))))))*((((-(r)))*(-1))))));
    double __diff___statement_2_t = __temp_78;
    double __temp_87 = ((K)*(((std::exp((((-(r)))*(((T)-(t))))))*((-(r))))));
    double __diff___statement_2_T = __temp_87;
    double __temp_96 = ((K)*(((std::exp((((-(r)))*(((T)-(t))))))*((((-(1)))*(((T)-(t))))))));
    double __diff___statement_2_r = __temp_96;
    double __diff___statement_2_S = 0.0;
    double __temp_114 = std::exp((((-(r)))*(((T)-(t)))));
    double __diff___statement_2_K = __temp_114;
    double __diff___statement_2_vol = 0.0;



    double __statement_3 = ((((std::erfc(-(__statement_0)/std::sqrt(2))/2)*(S)))-(((std::erfc(-(__statement_1)/std::sqrt(2))/2)*(__statement_2))));
    double __temp_132 = ((((((std::exp((-(((0.5)*(std::pow(__statement_0, 2)))))))/(2.50663)))*(S)))*(__diff___statement_0_t));
    double __temp_133 = (((-(((((std::exp((-(((0.5)*(std::pow(__statement_1, 2)))))))/(2.50663)))*(__statement_2)))))*(__diff___statement_1_t));
    double __temp_134 = (((-(std::erfc(-(__statement_1)/std::sqrt(2))/2)))*(__diff___statement_2_t));
    double __diff___statement_3_t = __temp_132 + __temp_133 + __temp_134;
    double __temp_141 = ((((((std::exp((-(((0.5)*(std::pow(__statement_0, 2)))))))/(2.50663)))*(S)))*(__diff___statement_0_T));
    double __temp_142 = (((-(((((std::exp((-(((0.5)*(std::pow(__statement_1, 2)))))))/(2.50663)))*(__statement_2)))))*(__diff___statement_1_T));
    double __temp_143 = (((-(std::erfc(-(__statement_1)/std::sqrt(2))/2)))*(__diff___statement_2_T));
    double __diff___statement_3_T = __temp_141 + __temp_142 + __temp_143;
    double __temp_150 = ((((((std::exp((-(((0.5)*(std::pow(__statement_0, 2)))))))/(2.50663)))*(S)))*(__diff___statement_0_r));
    double __temp_151 = (((-(((((std::exp((-(((0.5)*(std::pow(__statement_1, 2)))))))/(2.50663)))*(__statement_2)))))*(__diff___statement_1_r));
    double __temp_152 = (((-(std::erfc(-(__statement_1)/std::sqrt(2))/2)))*(__diff___statement_2_r));
    double __diff___statement_3_r = __temp_150 + __temp_151 + __temp_152;
    double __temp_156 = std::erfc(-(__statement_0)/std::sqrt(2))/2;
    double __temp_159 = ((((((std::exp((-(((0.5)*(std::pow(__statement_0, 2)))))))/(2.50663)))*(S)))*(__diff___statement_0_S));
    double __temp_160 = (((-(((((std::exp((-(((0.5)*(std::pow(__statement_1, 2)))))))/(2.50663)))*(__statement_2)))))*(__diff___statement_1_S));
    double __temp_161 = (((-(std::erfc(-(__statement_1)/std::sqrt(2))/2)))*(__diff___statement_2_S));
    double __diff___statement_3_S = __temp_156 + __temp_159 + __temp_160 + __temp_161;
    double __temp_168 = ((((((std::exp((-(((0.5)*(std::pow(__statement_0, 2)))))))/(2.50663)))*(S)))*(__diff___statement_0_K));
    double __temp_169 = (((-(((((std::exp((-(((0.5)*(std::pow(__statement_1, 2)))))))/(2.50663)))*(__statement_2)))))*(__diff___statement_1_K));
    double __temp_170 = (((-(std::erfc(-(__statement_1)/std::sqrt(2))/2)))*(__diff___statement_2_K));
    double __diff___statement_3_K = __temp_168 + __temp_169 + __temp_170;
    double __temp_177 = ((((((std::exp((-(((0.5)*(std::pow(__statement_0, 2)))))))/(2.50663)))*(S)))*(__diff___statement_0_vol));
    double __temp_178 = (((-(((((std::exp((-(((0.5)*(std::pow(__statement_1, 2)))))))/(2.50663)))*(__statement_2)))))*(__diff___statement_1_vol));
    double __temp_179 = (((-(std::erfc(-(__statement_1)/std::sqrt(2))/2)))*(__diff___statement_2_vol));
    double __diff___statement_3_vol = __temp_177 + __temp_178 + __temp_179;



    *d_t = __diff___statement_3_t;
    *d_T = __diff___statement_3_T;
    *d_r = __diff___statement_3_r;
    *d_S = __diff___statement_3_S;
    *d_K = __diff___statement_3_K;
    *d_vol = __diff___statement_3_vol;
    return __statement_3;
}

double black_raw(double t, double T, double r, double S, double K, double vol)
{
    double __statement_0 = ((((1)/(((vol)*(std::pow(((T)-(t)), 0.5))))))*(((std::log(((S)/(K))))+(((((r)+(((std::pow(vol, 2))/(2)))))*(((T)-(t))))))));
    double __statement_1 = ((__statement_0)-(((vol)*(((T)-(t))))));
    double __statement_2 = ((K)*(std::exp((((-(r)))*(((T)-(t)))))));
    double __statement_3 = ((((std::erfc(-(__statement_0)/std::sqrt(2))/2)*(S)))-(((std::erfc(-(__statement_1)/std::sqrt(2))/2)*(__statement_2))));
    return __statement_3;
}


double black_fd(double epsilon, double t, double d_t, double T, double d_T, double r, double d_r, double S, double d_S, double K, double d_K, double vol, double d_vol){
        double dummy;
        double lower = black_raw( t - d_t*epsilon/2 , T - d_T*epsilon/2  , r - d_r*epsilon/2  , S - d_S*epsilon/2  , K - d_K*epsilon/2  , vol - d_vol*epsilon/2);
        double upper = black_raw( t + d_t*epsilon/2 , T + d_T*epsilon/2  , r + d_r*epsilon/2  , S + d_S*epsilon/2  , K + d_K*epsilon/2  , vol + d_vol*epsilon/2);
        double finite_diff = ( upper - lower ) / epsilon;
        return finite_diff;
}
int main(){
        double t   = 0.0;
        double T   = 10.0;
        double r   = 0.04;
        double S   = 50;
        double K   = 60;
        double vol = 0.2;

        double epsilon = 1e-10;

        double d_t = 0.0;
        double d_T = 0.0;
        double d_r = 0.0;
        double d_S = 0.0;
        double d_K = 0.0;
        double d_vol = 0.0;
        double value = black( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);

        double d1 = 1/ ( vol * std::sqrt(T - t)) *  ( std::log(S/K) + ( r + vol*vol/2)*(T-t));

        double dummy;
        double lower = black( t - epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double upper = black( t + epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double finite_diff = ( upper - lower ) / epsilon;
        double residue = d_t - finite_diff;

        printf("%f,%f,%f,%f,%f,%f => %f,%f => %f,%f,%f\n", t, T, r, S, K, vol, value, d1, d_t, finite_diff, residue);

        printf("d[t]  ,%f,%f\n", d_t  ,  black_fd(epsilon, t, 1, T  , 0, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[T]  ,%f,%f\n", d_T  ,  black_fd(epsilon, t, 0, T  , 1, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[r]  ,%f,%f\n", d_r  ,  black_fd(epsilon, t, 0, T  , 0, r  , 1, S  , 0, K  , 0, vol, 0));
        printf("d[S]  ,%f,%f\n", d_S  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 1, K  , 0, vol, 0));
        printf("d[K]  ,%f,%f\n", d_K  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 1, vol, 0));
        printf("d[vol],%f,%f\n", d_vol,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 0, vol, 1));
        
        // time profile
        for(volatile size_t N = 100;;N*=2){
                boost::timer::cpu_timer timer;
                for(volatile size_t idx=0;idx!=N;++idx){
                        double value = black( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);
                }
                std::string ad_time = timer.format(4, "%w");
                timer.start();
                for(volatile size_t idx=0;idx!=N;++idx){
                        black_fd(epsilon, t, 1, T  , 0, r  , 0, S  , 0, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 1, r  , 0, S  , 0, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 1, S  , 0, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 1, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 1, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 0, vol, 1);
                }
                std::string fd_time = timer.format(4, "%w");
                std::cout << N << "," << fd_time << "," << ad_time << "\n";
        }

}
