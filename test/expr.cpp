#include <gtest/gtest.h>
#include "Cady/Cady.h"

using namespace Cady;

TEST(Expr,Constant){
        auto value = 1.234;
        auto x = Constant::Make(value);
        auto dx = x->Diff("x");

        EXPECT_EQ(OPKind_Constant, x->Kind()  );
        EXPECT_EQ(value, std::reinterpret_pointer_cast<Constant>(x)->Value() );
        EXPECT_EQ(OPKind_Constant, dx->Kind()  );
        EXPECT_EQ(0.0, std::reinterpret_pointer_cast<Constant>(dx)->Value() );
}

TEST(Expr,ExogenousSymbol){
        auto sym = "Foo";
        auto s = ExogenousSymbol::Make(sym);
        EXPECT_EQ(OPKind_ExogenousSymbol, s->Kind()  );
        EXPECT_EQ(sym, std::reinterpret_pointer_cast<ExogenousSymbol>(s)->Name() );

        auto ds_ds = s->Diff(sym);
        auto ds_dt = s->Diff("Other");

        EXPECT_EQ(OPKind_Constant, ds_ds->Kind());
        EXPECT_EQ(1.0, std::reinterpret_pointer_cast<Constant>(ds_ds)->Value() );
        EXPECT_EQ(OPKind_Constant, ds_dt->Kind());
        EXPECT_EQ(0.0, std::reinterpret_pointer_cast<Constant>(ds_dt)->Value() );
}

/*
        From Savine
 */
TEST(Black,Numeric){
        auto S0  = ExogenousSymbol::Make("S0");
        auto r   = ExogenousSymbol::Make("r");
        auto y   = ExogenousSymbol::Make("y");
        auto sig = ExogenousSymbol::Make("sig");
        auto K   = ExogenousSymbol::Make("K");
        auto T   = ExogenousSymbol::Make("T");

        auto df = Exp::Make( 
                UnaryOperator::UnaryMinus(
                        BinaryOperator::Mul(
                                r,
                                T
                        )
                )
        );

        auto F = BinaryOperator::Mul(
                S0,
                Exp::Make(
                        BinaryOperator::Mul(
                                BinaryOperator::Sub(r,y),
                                T
                        )
                )
        );

        auto std = BinaryOperator::Mul(
                sig,
                BinaryOperator::Pow(
                        T,
                        Constant::Make(0.5)
                )
        );

        auto d = BinaryOperator::Div(
                Log::Make(
                        BinaryOperator::Div(
                                EndgenousSymbol::Make("F", F),
                                EndgenousSymbol::Make("K", K)
                        )
                ),
                std
        );

        auto d1 = BinaryOperator::Add(
                EndgenousSymbol::Make("d", d),
                BinaryOperator::Mul(
                        Constant::Make(0.5),
                        EndgenousSymbol::Make("std", std)
                )
        );
        
        auto d2 = BinaryOperator::Sub(
                EndgenousSymbol::Make("d", d),
                BinaryOperator::Mul(
                        Constant::Make(0.5),
                        EndgenousSymbol::Make("std", std)
                )
        );

        auto nd1 = Phi::Make(EndgenousSymbol::Make("d1", d1));
        
        auto nd2 = Phi::Make(EndgenousSymbol::Make("d2", d2));

        auto c = BinaryOperator::Mul(
                EndgenousSymbol::Make("df", df),
                BinaryOperator::Sub(
                        BinaryOperator::Mul(
                                EndgenousSymbol::Make("F", F),
                                EndgenousSymbol::Make("nd1", nd1)
                        ),
                        BinaryOperator::Mul(
                                EndgenousSymbol::Make("K", K),
                                EndgenousSymbol::Make("nd2", nd2)
                        )
                )
        );

        double S0_value = 10;
        double r_value = 0.05;
        double y_value = 0.00;
        double sig_value = 0.2;
        double K_value = 13;
        double T_value = 1;

        SymbolTable ST;
        ST("S0" , S0_value);
        ST("r"  , r_value);
        ST("y"  , y_value);
        ST("sig", sig_value);
        ST("K"  , K_value);
        ST("T"  , T_value);

        std::cout << "c->Eval(ST) => " << c->Eval(ST) << "\n"; // __CandyPrint__(cxx-print-scalar,c->Eval(ST))

        using std::exp;
        using std::sqrt;
        auto phi = [](auto x){
                return std::erfc(-x/std::sqrt(2))/2;
        };

        double df_value = exp(-r_value*T_value);
        EXPECT_FLOAT_EQ(df_value, df->Eval(ST));

        double F_value = S0_value * exp((r_value-y_value)*T_value);
        EXPECT_FLOAT_EQ(F_value, F->Eval(ST));

        double std_value = sig_value * sqrt(T_value);
        EXPECT_FLOAT_EQ(std_value, std->Eval(ST));
        
        double d_value = log(F_value/K_value)/std_value;
        EXPECT_FLOAT_EQ(d_value, d->Eval(ST));

        double d1_value = d_value + 0.5 * std_value;
        EXPECT_FLOAT_EQ(d1_value, d1->Eval(ST));

        double d2_value = d_value - 0.5 * std_value;
        EXPECT_FLOAT_EQ(d2_value, d2->Eval(ST));

        double nd1_value = phi(d1_value);
        EXPECT_FLOAT_EQ(nd1_value, nd1->Eval(ST));

        double nd2_value = phi(d2_value);
        EXPECT_FLOAT_EQ(nd2_value, nd2->Eval(ST));

        double c_value = df_value * ( F_value * nd1_value - K_value * nd2_value );
        EXPECT_FLOAT_EQ(c_value, c->Eval(ST));

                                
                        
}

