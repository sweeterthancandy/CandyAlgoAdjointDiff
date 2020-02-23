sweeterthancady => candy algorithmic differentiation
====================================================

The idea of this is to have an in-memory expression tree which can be used to generate C++/C code

Scope is to be double only, with control flow statements

```c++
struct BlackScholesCallOption{
        template<class Double>
        struct Build{
                Double Evaluate(
                        Double t,
                        Double T,
                        Double r,
                        Double S,
                        Double K,
                        Double vol )const
                {
                        using MathFunctions::Phi;
                        using MathFunctions::Exp;
                        using MathFunctions::Pow;
                        using MathFunctions::Log;

                        Double d1 = ((1.0 / ( vol * Pow((T - t),0.5) )) * ( Log(S / K) +   (r + ( Pow(vol,2.0) ) / 2 ) * (T - t) ));
                        Double d2 = d1 - vol * (T - t);
                        Double pv = K * Exp( -r * ( T - t ) );
                        Double black = Phi(d1) * S - Phi(d2) * pv;
                        return black;
                }
        };
};
```


From this, I can both evaulate in code, or with an embedded dsl


```c++
double black(double t, double* d_t, double T, double* d_T, double r, double* d_r, double S, double* d_S, double K, double* d_K, double vol, double* d_vol){
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
```


How fast is it? I've got a file generated/black_0.cxx, with some profiles. initial results showned that the compiler can not simply be given neaive expressions (duplicate subexpression), and in-memory optimization is needed to beat finite difference


N|fd|ad|ad hand opt|ad three address
---|------|------|------|------
100|0.0001|0.0001|0.0000|0.0000
200|0.0003|0.0001|0.0000|0.0000
400|0.0005|0.0002|0.0001|0.0001
800|0.0010|0.0004|0.0001|0.0002
1600|0.0020|0.0009|0.0003|0.0003
3200|0.0018|0.0018|0.0005|0.0004
6400|0.0037|0.0016|0.0005|0.0006
12800|0.0073|0.0032|0.0010|0.0011
25600|0.0146|0.0064|0.0020|0.0022
51200|0.0293|0.0129|0.0039|0.0045
102400|0.0586|0.0257|0.0079|0.0090
204800|0.1171|0.0515|0.0157|0.0179
409600|0.2344|0.1030|0.0314|0.0358
819200|0.4688|0.2060|0.0628|0.0716
1638400|0.9374|0.4120|0.1257|0.1432
3276800|1.8742|0.8240|0.2521|0.2865
6553600|3.7657|1.6490|0.5028|0.5731
13107200|7.5069|3.2970|1.0054|1.1459
26214400|15.1139|6.6125|2.0190|2.3016
52428800|30.0545|13.4278|4.0232|4.6103
