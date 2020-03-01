
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


double black_manual_opt(double t, double* d_t, double T, double* d_T, double r, double* d_r, double S, double* d_S, double K, double* d_K, double vol, double* d_vol)
{
    auto time_to_expiry = (T)-(t);
    auto sk_ratio = (S)/(K);
    auto pow_1 = std::pow((time_to_expiry), 0.5);
    auto log_0 = std::log((sk_ratio));
    auto expr_10 = (vol)*(pow_1);
    auto pow_3 = std::pow((expr_10), 2);
    auto pow_4 = 1.0 / pow_1;
    auto pow_5 = std::pow(vol, 2);
    auto pow_6 = std::pow(K, 2);
    auto exp_2 = std::exp((((-(r)))*((time_to_expiry))));

    double __statement_0            = ((((1)/((expr_10))))*(((log_0)+(((((r)+(((pow_5)/(2)))))*((time_to_expiry)))))));
    double __temp_0                 = (((((((-(((vol)*(((0.5)*(((pow_4)*(-1)))))))))/(pow_3)))*(((log_0)+(((((r)+(((pow_5)/(2)))))*((time_to_expiry))))))))+(((((1)/((expr_10))))*(((((r)+(((pow_5)/(2)))))*(-1))))));
    double __diff___statement_0_t   = __temp_0;
    double __temp_7                 = (((((((-(((vol)*(((0.5)*(pow_4)))))))/(pow_3)))*(((log_0)+(((((r)+(((pow_5)/(2)))))*((time_to_expiry))))))))+(((((1)/((expr_10))))*(((r)+(((pow_5)/(2))))))));
    double __diff___statement_0_T   = __temp_7;
    double __temp_14                = ((((1)/((expr_10))))*((time_to_expiry)));
    double __diff___statement_0_r   = __temp_14;
    double __temp_21                = ((((1)/((expr_10))))*(((((K)/(pow_6)))/((sk_ratio)))));
    double __diff___statement_0_S   = __temp_21;
    double __temp_28                = ((((1)/((expr_10))))*((((((-(S)))/(pow_6)))/((sk_ratio)))));
    double __diff___statement_0_K   = __temp_28;
    double __temp_35                = (((((((-(pow_1)))/(pow_3)))*(((log_0)+(((((r)+(((pow_5)/(2)))))*((time_to_expiry))))))))+(((((1)/((expr_10))))*(((((((((2)*(vol)))*(2)))/(4)))*((time_to_expiry)))))));
    double __diff___statement_0_vol = __temp_35;
    
    auto exp_0                      = std::exp((-(((0.5)*(std::pow(__statement_0, 2))))));



    double __statement_1            = ((__statement_0)-(((vol)*((time_to_expiry)))));
    double __temp_36                = (-(((vol)*(-1))));
    double __temp_42                = __diff___statement_0_t;
    double __diff___statement_1_t   = __temp_36 + __temp_42;
    double __temp_44                = (-(vol));
    double __temp_49                = __diff___statement_0_T;
    double __diff___statement_1_T   = __temp_44 + __temp_49;
    double __temp_56                = __diff___statement_0_r;
    double __diff___statement_1_r   = __temp_56;
    double __temp_63                = __diff___statement_0_S;
    double __diff___statement_1_S   = __temp_63;
    double __temp_70                = __diff___statement_0_K;
    double __diff___statement_1_K   = __temp_70;
    double __temp_76                = (-((time_to_expiry)));
    double __temp_77                = __diff___statement_0_vol;
    double __diff___statement_1_vol = __temp_76 + __temp_77;

    auto pow_0                      = std::pow(__statement_1, 2);
    auto erfc_0                     = std::erfc(-(__statement_1)/std::sqrt(2))/2;
    auto expr_1                     = (((exp_0/(2.50663)))*(S));
    auto erfc_1                     = std::erfc(-(__statement_0)/std::sqrt(2))/2;


    double __statement_2            = ((K)*(exp_2));
    double __temp_78                = ((K)*(((exp_2)*((((-(r)))*(-1))))));
    double __diff___statement_2_t   = __temp_78;
    double __temp_87                = ((K)*(((exp_2)*((-(r))))));
    double __diff___statement_2_T   = __temp_87;
    double __temp_96                = ((K)*(((exp_2)*((((-(1)))*((time_to_expiry)))))));
    double __diff___statement_2_r   = __temp_96;
    double __diff___statement_2_S   = 0.0;
    double __temp_114               = exp_2;
    double __diff___statement_2_K   = __temp_114;
    double __diff___statement_2_vol = 0.0;


    auto expr_0                     = ((-(((((std::exp((-(((0.5)*(pow_0))))))/(2.50663)))*(__statement_2)))));

    double __statement_3            = ((((erfc_1)*(S)))-(((erfc_0)*(__statement_2))));
    double __temp_132               = ((expr_1)*(__diff___statement_0_t));
    double __temp_133               = (expr_0*(__diff___statement_1_t));
    double __temp_134               = (((-(erfc_0)))*(__diff___statement_2_t));
    double __diff___statement_3_t   = __temp_132 + __temp_133 + __temp_134;
    double __temp_141               = ((expr_1)*(__diff___statement_0_T));
    double __temp_142               = (expr_0*(__diff___statement_1_T));
    double __temp_143               = (((-(erfc_0)))*(__diff___statement_2_T));
    double __diff___statement_3_T   = __temp_141 + __temp_142 + __temp_143;
    double __temp_150               = ((expr_1)*(__diff___statement_0_r));
    double __temp_151               = (expr_0*(__diff___statement_1_r));
    double __temp_152               = (((-(erfc_0)))*(__diff___statement_2_r));
    double __diff___statement_3_r   = __temp_150 + __temp_151 + __temp_152;
    double __temp_156               = erfc_1;
    double __temp_159               = ((expr_1)*(__diff___statement_0_S));
    double __temp_160               = (expr_0*(__diff___statement_1_S));
    double __temp_161               = (((-(erfc_0)))*(__diff___statement_2_S));
    double __diff___statement_3_S   = __temp_156 + __temp_159 + __temp_160 + __temp_161;
    double __temp_168               = ((expr_1)*(__diff___statement_0_K));
    double __temp_169               = (expr_0*(__diff___statement_1_K));
    double __temp_170               = (((-(erfc_0)))*(__diff___statement_2_K));
    double __diff___statement_3_K   = __temp_168 + __temp_169 + __temp_170;
    double __temp_177               = ((expr_1)*(__diff___statement_0_vol));
    double __temp_178               = (expr_0*(__diff___statement_1_vol));
    double __temp_179               = (((-(erfc_0)))*(__diff___statement_2_vol));
    double __diff___statement_3_vol = __temp_177 + __temp_178 + __temp_179;



    *d_t                            = __diff___statement_3_t;
    *d_T                            = __diff___statement_3_T;
    *d_r                            = __diff___statement_3_r;
    *d_S                            = __diff___statement_3_S;
    *d_K                            = __diff___statement_3_K;
    *d_vol                          = __diff___statement_3_vol;
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



double black_better(double t, double* d_t, double T, double* d_T, double r, double* d_r, double S, double* d_S, double K, double* d_K, double vol, double* d_vol){
    double __symbol_5      = ((T)-(t));
    double __symbol_6      = std::pow(__symbol_5, 0.5);
    double __symbol_7      = ((vol)*(__symbol_6));
    double __symbol_8      = ((1)/(__symbol_7));
    double __symbol_11     = ((S)/(K));
    double __symbol_12     = std::log(__symbol_11);
    double __symbol_15     = std::pow(vol, 2);
    double __symbol_16     = ((__symbol_15)/(2));
    double __symbol_17     = ((r)+(__symbol_16));
    double __symbol_18     = ((__symbol_17)*(__symbol_5));
    double __symbol_19     = ((__symbol_12)+(__symbol_18));
    double __symbol_20     = ((__symbol_8)*(__symbol_19));
    double __symbol_21     = std::pow(__symbol_20, 2);
    double __symbol_22     = ((0.5)*(__symbol_21));
    double __symbol_23     = (-(__symbol_22));
    double __symbol_24     = std::exp(__symbol_23);
    double __symbol_26     = ((__symbol_24)/(2.50663));
    double __symbol_28     = std::pow(__symbol_5, -0.5);
    double __symbol_30     = ((__symbol_28)*(-1));
    double __symbol_31     = ((0.5)*(__symbol_30));
    double __symbol_32     = ((vol)*(__symbol_31));
    double __symbol_33     = (-(__symbol_32));
    double __symbol_34     = std::pow(__symbol_7, 2);
    double __symbol_35     = ((__symbol_33)/(__symbol_34));
    double __symbol_36     = ((__symbol_35)*(__symbol_19));
    double __symbol_37     = ((__symbol_17)*(-1));
    double __symbol_38     = ((__symbol_8)*(__symbol_37));
    double __symbol_39     = ((__symbol_36)+(__symbol_38));
    double __symbol_40     = ((__symbol_26)*(__symbol_39));
    double __symbol_41     = ((__symbol_40)*(S));
    double __symbol_42     = ((vol)*(__symbol_5));
    double __symbol_43     = ((__symbol_20)-(__symbol_42));
    double __symbol_44     = std::pow(__symbol_43, 2);
    double __symbol_45     = ((0.5)*(__symbol_44));
    double __symbol_46     = (-(__symbol_45));
    double __symbol_47     = std::exp(__symbol_46);
    double __symbol_48     = ((__symbol_47)/(2.50663));
    double __symbol_49     = ((vol)*(-1));
    double __symbol_50     = ((__symbol_39)-(__symbol_49));
    double __symbol_51     = ((__symbol_48)*(__symbol_50));
    double __symbol_52     = (-(r));
    double __symbol_53     = ((__symbol_52)*(__symbol_5));
    double __symbol_54     = std::exp(__symbol_53);
    double __symbol_55     = ((K)*(__symbol_54));
    double __symbol_56     = ((__symbol_51)*(__symbol_55));
    double __symbol_57     = std::erfc(-(__symbol_43)/std::sqrt(2))/2;
    double __symbol_58     = ((__symbol_52)*(-1));
    double __symbol_59     = ((__symbol_54)*(__symbol_58));
    double __symbol_60     = ((K)*(__symbol_59));
    double __symbol_61     = ((__symbol_57)*(__symbol_60));
    double __symbol_62     = ((__symbol_56)+(__symbol_61));
    *d_t = ((__symbol_41)-(__symbol_62));
    double __symbol_63     = ((0.5)*(__symbol_28));
    double __symbol_64     = ((vol)*(__symbol_63));
    double __symbol_65     = (-(__symbol_64));
    double __symbol_66     = ((__symbol_65)/(__symbol_34));
    double __symbol_67     = ((__symbol_66)*(__symbol_19));
    double __symbol_68     = ((__symbol_8)*(__symbol_17));
    double __symbol_69     = ((__symbol_67)+(__symbol_68));
    double __symbol_70     = ((__symbol_26)*(__symbol_69));
    double __symbol_71     = ((__symbol_70)*(S));
    double __symbol_72     = ((__symbol_69)-(vol));
    double __symbol_73     = ((__symbol_48)*(__symbol_72));
    double __symbol_74     = ((__symbol_73)*(__symbol_55));
    double __symbol_75     = ((__symbol_54)*(__symbol_52));
    double __symbol_76     = ((K)*(__symbol_75));
    double __symbol_77     = ((__symbol_57)*(__symbol_76));
    double __symbol_78     = ((__symbol_74)+(__symbol_77));
    *d_T = ((__symbol_71)-(__symbol_78));
    double __symbol_79     = ((__symbol_8)*(__symbol_5));
    double __symbol_80     = ((__symbol_26)*(__symbol_79));
    double __symbol_81     = ((__symbol_80)*(S));
    double __symbol_82     = ((__symbol_48)*(__symbol_79));
    double __symbol_83     = ((__symbol_82)*(__symbol_55));
    double __symbol_84     = (-(1));
    double __symbol_85     = ((__symbol_84)*(__symbol_5));
    double __symbol_86     = ((__symbol_54)*(__symbol_85));
    double __symbol_87     = ((K)*(__symbol_86));
    double __symbol_88     = ((__symbol_57)*(__symbol_87));
    double __symbol_89     = ((__symbol_83)+(__symbol_88));
    *d_r = ((__symbol_81)-(__symbol_89));
    double __symbol_90     = std::pow(K, 2);
    double __symbol_91     = ((K)/(__symbol_90));
    double __symbol_92     = ((__symbol_91)/(__symbol_11));
    double __symbol_93     = ((__symbol_8)*(__symbol_92));
    double __symbol_94     = ((__symbol_26)*(__symbol_93));
    double __symbol_95     = ((__symbol_94)*(S));
    double __symbol_96     = std::erfc(-(__symbol_20)/std::sqrt(2))/2;
    double __symbol_97     = ((__symbol_95)+(__symbol_96));
    double __symbol_98     = ((__symbol_48)*(__symbol_93));
    double __symbol_99     = ((__symbol_98)*(__symbol_55));
    *d_S = ((__symbol_97)-(__symbol_99));
    double __symbol_100    = (-(S));
    double __symbol_101    = ((__symbol_100)/(__symbol_90));
    double __symbol_102    = ((__symbol_101)/(__symbol_11));
    double __symbol_103    = ((__symbol_8)*(__symbol_102));
    double __symbol_104    = ((__symbol_26)*(__symbol_103));
    double __symbol_105    = ((__symbol_104)*(S));
    double __symbol_106    = ((__symbol_48)*(__symbol_103));
    double __symbol_107    = ((__symbol_106)*(__symbol_55));
    double __symbol_108    = ((__symbol_57)*(__symbol_54));
    double __symbol_109    = ((__symbol_107)+(__symbol_108));
    *d_K = ((__symbol_105)-(__symbol_109));
    double __symbol_110    = (-(__symbol_6));
    double __symbol_111    = ((__symbol_110)/(__symbol_34));
    double __symbol_112    = ((__symbol_111)*(__symbol_19));
    double __symbol_113    = ((2)*(vol));
    double __symbol_114    = ((__symbol_113)*(2));
    double __symbol_116    = ((__symbol_114)/(4));
    double __symbol_117    = ((__symbol_116)*(__symbol_5));
    double __symbol_118    = ((__symbol_8)*(__symbol_117));
    double __symbol_119    = ((__symbol_112)+(__symbol_118));
    double __symbol_120    = ((__symbol_26)*(__symbol_119));
    double __symbol_121    = ((__symbol_120)*(S));
    double __symbol_122    = ((__symbol_119)-(__symbol_5));
    double __symbol_123    = ((__symbol_48)*(__symbol_122));
    double __symbol_124    = ((__symbol_123)*(__symbol_55));
    *d_vol = ((__symbol_121)-(__symbol_124));
    double __symbol_125    = ((__symbol_96)*(S));
    double __symbol_126    = ((__symbol_57)*(__symbol_55));
    return ((__symbol_125)-(__symbol_126));
}

double black_backwards(double t, double* d_t, double T, double* d_T, double r, double* d_r, double S, double* d_S, double K, double* d_K, double vol, double* d_vol){
        double w2 = vol;
        double w3 = T;
        double w4 = t;
        double w5 = ((w3)-(w4));
        double w7 = std::pow(w5, 0.5);
        double w8 = ((w2)*(w7));
        double w9 = ((1)/(w8));
        double w10 = S;
        double w11 = K;
        double w12 = ((w10)/(w11));
        double w13 = std::log(w12);
        double w14 = r;
        double w16 = std::pow(w2, 2);
        double w17 = ((w16)/(2));
        double w18 = ((w14)+(w17));
        double w19 = ((w18)*(w5));
        double w20 = ((w13)+(w19));
        double w21 = ((w9)*(w20));
        double __statement_0 = w21;
        double w23 = std::erfc(-(__statement_0)/std::sqrt(2))/2;
        double w24 = ((w23)*(w10));
        double w25 = ((w2)*(w5));
        double w26 = ((__statement_0)-(w25));
        double __statement_1 = w26;
        double w28 = std::erfc(-(__statement_1)/std::sqrt(2))/2;
        double w29 = (-(w14));
        double w30 = ((w29)*(w5));
        double w31 = std::exp(w30);
        double w32 = ((w11)*(w31));
        double __statement_2 = w32;
        double w34 = ((w28)*(__statement_2));
        double w35 = ((w24)-(w34));
        double __rev_ad_w35 = 1.0;
        double __rev_ad_w34 = ((-1)*(__rev_ad_w35));
        double __rev_ad___statement_2 = ((w28)*(__rev_ad_w34));
        double __rev_ad_w32 = __rev_ad___statement_2;
        double __rev_ad_w31 = ((w11)*(__rev_ad_w32));
        double __rev_ad_w30 = ((std::exp(w30))*(__rev_ad_w31));
        double __rev_ad_w29 = ((w5)*(__rev_ad_w30));
        double __rev_ad_w28 = ((__statement_2)*(__rev_ad_w34));
        double __rev_ad___statement_1 = ((((std::exp((-(((0.5)*(std::pow(__statement_1, 2)))))))/(2.50663)))*(__rev_ad_w28));
        double __rev_ad_w26 = __rev_ad___statement_1;
        double __rev_ad_w25 = ((-1)*(__rev_ad_w26));
        double __rev_ad_w24 = __rev_ad_w35;
        double __rev_ad_w23 = ((w10)*(__rev_ad_w24));
        double __rev_ad___statement_0 = ((((((std::exp((-(((0.5)*(std::pow(__statement_0, 2)))))))/(2.50663)))*(__rev_ad_w23)))+(__rev_ad_w26));
        double __rev_ad_w21 = __rev_ad___statement_0;
        double __rev_ad_w20 = ((w9)*(__rev_ad_w21));
        double __rev_ad_w19 = __rev_ad_w20;
        double __rev_ad_w18 = ((w5)*(__rev_ad_w19));
        double __rev_ad_w17 = __rev_ad_w18;
        double __rev_ad_w16 = ((((2)/(4)))*(__rev_ad_w17));
        double __rev_ad_w14 = ((__rev_ad_w18)+((((-(1)))*(__rev_ad_w29))));
        *d_r = __rev_ad_w14;
        double __rev_ad_w13 = __rev_ad_w20;
        double __rev_ad_w12 = ((((1)/(w12)))*(__rev_ad_w13));
        double __rev_ad_w11 = (((((((-(w10)))/(std::pow(w11, 2))))*(__rev_ad_w12)))+(((w31)*(__rev_ad_w32))));
        *d_K = __rev_ad_w11;
        double __rev_ad_w10 = ((((((w11)/(std::pow(w11, 2))))*(__rev_ad_w12)))+(((w23)*(__rev_ad_w24))));
        *d_S = __rev_ad_w10;
        double __rev_ad_w9 = ((w20)*(__rev_ad_w21));
        double __rev_ad_w8 = ((((-1)/(std::pow(w8, 2))))*(__rev_ad_w9));
        double __rev_ad_w7 = ((w2)*(__rev_ad_w8));
        double __rev_ad_w5 = ((((((((((0.5)*(std::pow(w5, -0.5))))*(__rev_ad_w7)))+(((w18)*(__rev_ad_w19)))))+(((w2)*(__rev_ad_w25)))))+(((w29)*(__rev_ad_w30))));
        double __rev_ad_w4 = ((-1)*(__rev_ad_w5));
        *d_t = __rev_ad_w4;
        double __rev_ad_w3 = __rev_ad_w5;
        *d_T = __rev_ad_w3;
        double __rev_ad_w2 = ((((((w7)*(__rev_ad_w8)))+(((((2)*(w2)))*(__rev_ad_w16)))))+(((w5)*(__rev_ad_w25))));
        *d_vol = __rev_ad_w2;
        return w35;
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
        
        double d_t_opt = 0.0;
        double d_T_opt = 0.0;
        double d_r_opt = 0.0;
        double d_S_opt = 0.0;
        double d_K_opt = 0.0;
        double d_vol_opt = 0.0;
        double value_opt = black_manual_opt( t  , &d_t_opt, T  , &d_T_opt, r  , &d_r_opt, S  , &d_S_opt, K  , &d_K_opt, vol, &d_vol_opt);
        
        double d_t_better = 0.0;
        double d_T_better = 0.0;
        double d_r_better = 0.0;
        double d_S_better = 0.0;
        double d_K_better = 0.0;
        double d_vol_better = 0.0;
        double value_better = black_better( t  , &d_t_better, T  , &d_T_better, r  , &d_r_better, S  , &d_S_better, K  , &d_K_better, vol, &d_vol_better);
        
        double d_t_backwards = 0.0;
        double d_T_backwards = 0.0;
        double d_r_backwards = 0.0;
        double d_S_backwards = 0.0;
        double d_K_backwards = 0.0;
        double d_vol_backwards = 0.0;
        double value_backwards = black_backwards( t  , &d_t_backwards, T  , &d_T_backwards, r  , &d_r_backwards, S  , &d_S_backwards, K  , &d_K_backwards, vol, &d_vol_backwards);


        printf("call  ,%f,%f,%f,%f\n"     , value, value_opt, value_better, value_backwards);
        printf("d[t]  ,%f,%f,%f,%f,%f\n"  , d_t      , d_t_opt  , d_t_better  , d_t_backwards  ,black_fd(epsilon, t, 1, T  , 0, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[T]  ,%f,%f,%f,%f,%f\n"  , d_T      , d_T_opt  , d_T_better  , d_T_backwards  ,black_fd(epsilon, t, 0, T  , 1, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[r]  ,%f,%f,%f,%f,%f\n"  , d_r      , d_r_opt  , d_r_better  , d_r_backwards  ,black_fd(epsilon, t, 0, T  , 0, r  , 1, S  , 0, K  , 0, vol, 0));
        printf("d[S]  ,%f,%f,%f,%f,%f\n"  , d_S      , d_S_opt  , d_S_better  , d_S_backwards  ,black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 1, K  , 0, vol, 0));
        printf("d[K]  ,%f,%f,%f,%f,%f\n"  , d_K      , d_K_opt  , d_K_better  , d_K_backwards  ,black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 1, vol, 0));
        printf("d[vol],%f,%f,%f,%f,%f\n"  , d_vol    , d_vol_opt, d_vol_better, d_vol_backwards,black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 0, vol, 1));
        
        // time profile
        for(volatile size_t N = 100;;N*=2){
                boost::timer::cpu_timer timer;
                for(volatile size_t idx=0;idx!=N;++idx){
                        double value = black( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);
                }
                std::string ad_time = timer.format(4, "%w");
                timer.start();

                for(volatile size_t idx=0;idx!=N;++idx){
                        double value = black_manual_opt( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);
                }
                std::string ad_opt_time = timer.format(4, "%w");
                timer.start();
                
                for(volatile size_t idx=0;idx!=N;++idx){
                        double value = black_better( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);
                }
                std::string ad_better = timer.format(4, "%w");
                timer.start();
                
                for(volatile size_t idx=0;idx!=N;++idx){
                        double value = black_backwards( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);
                }
                std::string ad_backwards = timer.format(4, "%w");
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
                std::cout << N << "," << fd_time << "," << ad_time << "," << ad_opt_time << "," << ad_better << "," << ad_backwards << "\n";
        }

}
