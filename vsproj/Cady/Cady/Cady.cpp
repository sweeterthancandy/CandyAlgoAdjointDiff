#include <iostream>


#include "Cady/Frontend.h"
#include "Cady/Transform.h"
#include "Cady/Cady.h"
#include "Cady/SymbolicMatrix.h"
#include "Cady/Instruction.h"
#include "Cady/ImpliedMatrixFunction.h"
#include "Cady/AADFunctionGenerator.h"
#include "Cady/CodeWriter.h"
#include "Cady/ProgramCode.h"
#include "Cady/CodeGen.h"
#include "Cady/CpuTimer.h"


#include <array>
#include <map>
#include <functional>

#include "Cady/Templates/CallOption.h"
#include "Cady/Templates/KOCallOption.h"

using namespace Cady;
using namespace Cady::CodeGen;






int main()
{
    auto w = ModuleWriter("C:\\temp\\Generated.h");
    w.EmitModule< KoBarrierOption::Module>();

#if 0

    using kernel_ty = KoBarrierOption::KoBarrierCallOption;

    /*
    * Double x, 
                Double K,
                Double tau,
                Double r,
                Double sigma,
                Double B
    */
    double S = 80;
    double K = 100;
    double tau = 1.0;
    double r = 0.00;
    double vol = 0.2;
    double B = 120;
    std::cout << kernel_ty::Build<double>{}.Evaluate(S, K, tau, r, vol, B) << "\n";
    auto aad_result = __KoBarrierCallOption(S, K, tau, r, vol, B);



    std::vector<double> X{ S, K, tau, r, vol, B };
    std::cout << kernel_ty::Build<double>{}.EvaluateVec(X) << "\n";
    std::cout << "numeric=" << kernel_ty::Build<double>{}.EvaluateVec(X) << ", AAD=" << aad_result[0] << "\n";
    for (size_t idx = 0; idx != X.size(); ++idx)
    {
        const double epsilon = 1e-8;
        const double old_value = X[idx];
        X[idx] = old_value - epsilon / 2;
        auto left = kernel_ty::Build<double>{}.EvaluateVec(X);
        X[idx] = old_value + epsilon / 2;
        auto right = kernel_ty::Build<double>{}.EvaluateVec(X);

        auto numeric = (right - left) / epsilon;

        std::cout << "numeric=" << numeric << ", AAD=" << aad_result[idx + 1] << "\n";
    }

    size_t num_evals = 10000000;
    {
        cpu_timer timer;
        for (volatile size_t idx = 0; idx != num_evals; ++idx)
            kernel_ty::Build<double>{}.Evaluate(S, K, tau, r, vol, B);
        std::cout << std::setw(30) << "RAW function" << " => " << timer.format() << "\n";
    }
    {
        cpu_timer timer;
        for (volatile size_t idx = 0; idx != num_evals; ++idx)
            __KoBarrierCallOption(S, K, tau, r, vol, B);
        std::cout << std::setw(30) << "AAD function" << " => " << timer.format() << "\n";
    }
#endif
}
