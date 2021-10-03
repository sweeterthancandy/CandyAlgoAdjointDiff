double BlackScholesCallOptionTestBareMetal(double t, double T, double r, double S, double K, double vol, double* d_t = nullptr, double* d_T = nullptr, double* d_r = nullptr, double* d_S = nullptr, double* d_K = nullptr, double* d_vol = nullptr)
{
    // Matrix function F_0 => __symbol_9
    //     r => 1.000000

    double const __symbol_9 = r;

    // Matrix function F_1 => __symbol_30
    //     __symbol_9 => (-(1.000000))

    double const __symbol_30 = (-(__symbol_9));

    // Matrix function F_2 => __symbol_2
    //     T => 1.000000

    double const __symbol_2 = T;

    // Matrix function F_3 => __symbol_31
    //     __symbol_30 => __symbol_2
    //     __symbol_2 => __symbol_30

    double const __symbol_31 = ((__symbol_30)*(__symbol_2));

    // Matrix function F_4 => __symbol_32
    //     __symbol_31 => std::exp(__symbol_31)

    double const __symbol_32 = std::exp(__symbol_31);

    // Matrix function F_5 => __statement_9
    //     __symbol_32 => 1.000000

    double const __statement_9 = __symbol_32;

    // Matrix function F_6 => __symbol_12
    //     S => 1.000000

    double const __symbol_12 = S;

    // Matrix function F_7 => __symbol_10
    //     __symbol_9 => __symbol_2
    //     __symbol_2 => __symbol_9

    double const __symbol_10 = ((__symbol_9)*(__symbol_2));

    // Matrix function F_8 => __symbol_11
    //     __symbol_10 => std::exp(__symbol_10)

    double const __symbol_11 = std::exp(__symbol_10);

    // Matrix function F_9 => __symbol_13
    //     __symbol_12 => __symbol_11
    //     __symbol_11 => __symbol_12

    double const __symbol_13 = ((__symbol_12)*(__symbol_11));

    // Matrix function F_10 => __statement_10
    //     __symbol_13 => 1.000000

    double const __statement_10 = __symbol_13;

    // Matrix function F_11 => __symbol_8
    //     K => 1.000000

    double const __symbol_8 = K;

    // Matrix function F_12 => __symbol_15
    //     __statement_10 => ((__symbol_8)/(std::pow(__symbol_8, 2.000000)))
    //     __symbol_8 => (((-(__statement_10)))/(std::pow(__symbol_8, 2.000000)))

    double const __symbol_15 = ((__statement_10)/(__symbol_8));

    // Matrix function F_13 => __symbol_16
    //     __symbol_15 => ((1.000000)/(__symbol_15))

    double const __symbol_16 = std::log(__symbol_15);

    // Matrix function F_14 => __symbol_4
    //     vol => 1.000000

    double const __symbol_4 = vol;

    // Matrix function F_15 => __symbol_3
    //     __symbol_2 => ((0.500000)*(std::pow(__symbol_2, -0.500000)))

    double const __symbol_3 = std::pow(__symbol_2, 0.500000);

    // Matrix function F_16 => __symbol_5
    //     __symbol_4 => __symbol_3
    //     __symbol_3 => __symbol_4

    double const __symbol_5 = ((__symbol_4)*(__symbol_3));

    // Matrix function F_17 => __statement_11
    //     __symbol_5 => 1.000000

    double const __statement_11 = __symbol_5;

    // Matrix function F_18 => __symbol_17
    //     __statement_11 => (((-(__symbol_16)))/(std::pow(__statement_11, 2.000000)))
    //     __symbol_16 => ((__statement_11)/(std::pow(__statement_11, 2.000000)))

    double const __symbol_17 = ((__symbol_16)/(__statement_11));

    // Matrix function F_19 => __statement_12
    //     __symbol_17 => 1.000000

    double const __statement_12 = __symbol_17;

    // Matrix function F_20 => __symbol_7
    //     __statement_11 => 0.500000

    double const __symbol_7 = ((0.500000)*(__statement_11));

    // Matrix function F_21 => __symbol_19
    //     __statement_12 => 1.000000
    //     __symbol_7 => 1.000000

    double const __symbol_19 = ((__statement_12)+(__symbol_7));

    // Matrix function F_22 => __statement_13
    //     __symbol_19 => 1.000000

    double const __statement_13 = __symbol_19;

    // Matrix function F_23 => __symbol_26
    //     __statement_13 => ((std::exp((-(((0.500000)*(std::pow(__statement_13, 2.000000)))))))/(2.506628))

    double const __symbol_26 = std::erfc(-(__statement_13)/std::sqrt(2))/2;

    // Matrix function F_24 => __statement_15
    //     __symbol_26 => 1.000000

    double const __statement_15 = __symbol_26;

    // Matrix function F_25 => __symbol_28
    //     __statement_10 => __statement_15
    //     __statement_15 => __statement_10

    double const __symbol_28 = ((__statement_10)*(__statement_15));

    // Matrix function F_26 => __symbol_21
    //     __statement_13 => 1.000000
    //     __statement_11 => -1.000000

    double const __symbol_21 = ((__statement_13)-(__statement_11));

    // Matrix function F_27 => __statement_14
    //     __symbol_21 => 1.000000

    double const __statement_14 = __symbol_21;

    // Matrix function F_28 => __symbol_23
    //     __statement_14 => ((std::exp((-(((0.500000)*(std::pow(__statement_14, 2.000000)))))))/(2.506628))

    double const __symbol_23 = std::erfc(-(__statement_14)/std::sqrt(2))/2;

    // Matrix function F_29 => __statement_16
    //     __symbol_23 => 1.000000

    double const __statement_16 = __symbol_23;

    // Matrix function F_30 => __symbol_25
    //     __symbol_8 => __statement_16
    //     __statement_16 => __symbol_8

    double const __symbol_25 = ((__symbol_8)*(__statement_16));

    // Matrix function F_31 => __symbol_29
    //     __symbol_25 => -1.000000
    //     __symbol_28 => 1.000000

    double const __symbol_29 = ((__symbol_28)-(__symbol_25));

    // Matrix function F_32 => __symbol_34
    //     __symbol_29 => __statement_9
    //     __statement_9 => __symbol_29

    double const __symbol_34 = ((__statement_9)*(__symbol_29));

    // Matrix function F_33 => __statement_17
    //     __symbol_34 => 1.000000

    double const __statement_17 = __symbol_34;

    // // AD section

    double const result_d_t = 0.000000;

    if( d_t) { *d_t = result_d_t; }

    double const __symbol_130 = std::pow(__symbol_2, -0.500000);

    double const __symbol_131 = ((0.500000)*(__symbol_130));

    double const __symbol_46 = ((__symbol_9)*(__symbol_2));

    double const __symbol_48 = std::exp(__symbol_10);

    double const __symbol_51 = ((__symbol_12)*(__symbol_11));

    double const __symbol_54 = ((__statement_10)/(__symbol_8));

    double const __symbol_56 = std::log(__symbol_15);

    double const __symbol_36 = std::pow(__symbol_2, 0.500000);

    double const __symbol_39 = ((__symbol_4)*(__symbol_3));

    double const __symbol_58 = ((__symbol_16)/(__statement_11));

    double const __symbol_42 = ((0.500000)*(__statement_11));

    double const __symbol_61 = ((__statement_12)+(__symbol_7));

    double const __symbol_64 = ((__statement_13)-(__statement_11));

    double const __symbol_100 = std::pow(__statement_14, 2.000000);

    double const __symbol_101 = ((0.500000)*(__symbol_100));

    double const __symbol_102 = (-(__symbol_101));

    double const __symbol_103 = std::exp(__symbol_102);

    double const __symbol_104 = ((__symbol_103)/(2.506628));

    double const __symbol_79 = (-(__symbol_9));

    double const __symbol_81 = ((__symbol_30)*(__symbol_2));

    double const __symbol_83 = std::exp(__symbol_31);

    double const __symbol_98 = ((-1.000000)*(__statement_9));

    double const __symbol_99 = ((__symbol_8)*(__symbol_98));

    double const __symbol_105 = ((__symbol_104)*(__symbol_99));

    double const __symbol_125 = ((-1.000000)*(__symbol_105));

    double const __symbol_91 = std::pow(__statement_13, 2.000000);

    double const __symbol_92 = ((0.500000)*(__symbol_91));

    double const __symbol_93 = (-(__symbol_92));

    double const __symbol_94 = std::exp(__symbol_93);

    double const __symbol_95 = ((__symbol_94)/(2.506628));

    double const __symbol_88 = ((__statement_10)*(__statement_9));

    double const __symbol_96 = ((__symbol_95)*(__symbol_88));

    double const __symbol_106 = ((__symbol_105)+(__symbol_96));

    double const __symbol_124 = ((0.500000)*(__symbol_106));

    double const __symbol_126 = ((__symbol_125)+(__symbol_124));

    double const __symbol_121 = (-(__symbol_16));

    double const __symbol_107 = std::pow(__statement_11, 2.000000);

    double const __symbol_122 = ((__symbol_121)/(__symbol_107));

    double const __symbol_123 = ((__symbol_122)*(__symbol_106));

    double const __symbol_127 = ((__symbol_126)+(__symbol_123));

    double const __symbol_128 = ((__symbol_4)*(__symbol_127));

    double const __symbol_132 = ((__symbol_131)*(__symbol_128));

    double const __symbol_72 = std::erfc(-(__statement_13)/std::sqrt(2))/2;

    double const __symbol_116 = ((__statement_15)*(__statement_9));

    double const __symbol_113 = std::pow(__symbol_8, 2.000000);

    double const __symbol_114 = ((__symbol_8)/(__symbol_113));

    double const __symbol_111 = ((1.000000)/(__symbol_15));

    double const __symbol_108 = ((__statement_11)/(__symbol_107));

    double const __symbol_109 = ((__symbol_108)*(__symbol_106));

    double const __symbol_112 = ((__symbol_111)*(__symbol_109));

    double const __symbol_115 = ((__symbol_114)*(__symbol_112));

    double const __symbol_117 = ((__symbol_116)+(__symbol_115));

    double const __symbol_118 = ((__symbol_12)*(__symbol_117));

    double const __symbol_119 = ((__symbol_48)*(__symbol_118));

    double const __symbol_120 = ((__symbol_9)*(__symbol_119));

    double const __symbol_133 = ((__symbol_132)+(__symbol_120));

    double const __symbol_75 = ((__statement_10)*(__statement_15));

    double const __symbol_67 = std::erfc(-(__statement_14)/std::sqrt(2))/2;

    double const __symbol_70 = ((__symbol_8)*(__statement_16));

    double const __symbol_77 = ((__symbol_28)-(__symbol_25));

    double const __symbol_84 = ((__symbol_83)*(__symbol_29));

    double const __symbol_85 = ((__symbol_30)*(__symbol_84));

    double const result_d_T = ((__symbol_133)+(__symbol_85));

    if( d_T) { *d_T = result_d_T; }

    double const __symbol_137 = ((__symbol_2)*(__symbol_119));

    double const __symbol_135 = (-(1.000000));

    double const __symbol_134 = ((__symbol_2)*(__symbol_84));

    double const __symbol_136 = ((__symbol_135)*(__symbol_134));

    double const result_d_r = ((__symbol_137)+(__symbol_136));

    if( d_r) { *d_r = result_d_r; }

    double const result_d_S = ((__symbol_11)*(__symbol_117));

    if( d_S) { *d_S = result_d_S; }

    double const __symbol_141 = ((__statement_16)*(__symbol_98));

    double const __symbol_138 = (-(__statement_10));

    double const __symbol_139 = ((__symbol_138)/(__symbol_113));

    double const __symbol_140 = ((__symbol_139)*(__symbol_112));

    double const result_d_K = ((__symbol_141)+(__symbol_140));

    if( d_K) { *d_K = result_d_K; }

    double const result_d_vol = ((__symbol_3)*(__symbol_127));

    if( d_vol) { *d_vol = result_d_vol; }

    return __statement_17;

}
double BlackScholesCallOptionTestBareMetalNoDiff(double t, double T, double r, double S, double K, double vol)
{
    double const __symbol_9 = r;

    double const __symbol_30 = (-(__symbol_9));

    double const __symbol_2 = T;

    double const __symbol_31 = ((__symbol_30)*(__symbol_2));

    double const __symbol_32 = std::exp(__symbol_31);

    double const __statement_0 = __symbol_32;

    double const __symbol_12 = S;

    double const __symbol_10 = ((__symbol_9)*(__symbol_2));

    double const __symbol_11 = std::exp(__symbol_10);

    double const __symbol_13 = ((__symbol_12)*(__symbol_11));

    double const __statement_1 = __symbol_13;

    double const __symbol_8 = K;

    double const __symbol_15 = ((__statement_1)/(__symbol_8));

    double const __symbol_16 = std::log(__symbol_15);

    double const __symbol_4 = vol;

    double const __symbol_3 = std::pow(__symbol_2, 0.500000);

    double const __symbol_5 = ((__symbol_4)*(__symbol_3));

    double const __statement_2 = __symbol_5;

    double const __symbol_17 = ((__symbol_16)/(__statement_2));

    double const __statement_3 = __symbol_17;

    double const __symbol_7 = ((0.500000)*(__statement_2));

    double const __symbol_19 = ((__statement_3)+(__symbol_7));

    double const __statement_4 = __symbol_19;

    double const __symbol_26 = std::erfc(-(__statement_4)/std::sqrt(2))/2;

    double const __statement_6 = __symbol_26;

    double const __symbol_28 = ((__statement_1)*(__statement_6));

    double const __symbol_21 = ((__statement_4)-(__statement_2));

    double const __statement_5 = __symbol_21;

    double const __symbol_23 = std::erfc(-(__statement_5)/std::sqrt(2))/2;

    double const __statement_7 = __symbol_23;

    double const __symbol_25 = ((__symbol_8)*(__statement_7));

    double const __symbol_29 = ((__symbol_28)-(__symbol_25));

    double const __symbol_34 = ((__statement_0)*(__symbol_29));

    double const __statement_8 = __symbol_34;

    return __statement_8;

}
