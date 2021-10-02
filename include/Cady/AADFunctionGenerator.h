#ifndef INCLUDE_ADD_FUNCTION_GENERATOR_H
#define INCLUDE_ADD_FUNCTION_GENERATOR_H

#include "Cady.h"

namespace Cady {

    template<class Function>
    class AADFunctionGenerator
    {
    public:
        std::shared_ptr<InstructionBlock> GenerateInstructionBlock()const
        {
            auto ad_kernel = Function::template Build<DoubleKernel>();

            auto arguments = ad_kernel.Arguments();

            std::vector<DoubleKernel> symbolc_arguments;
            for (auto const& arg : arguments)
            {
                symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
            }
            auto as_black = ad_kernel.EvaluateVec(symbolc_arguments);

            auto expr = as_black.as_operator_();
            auto RU = std::make_shared<Transform::RemapUnique>();
            auto unique = expr->Clone(RU);

            auto exo_symbols = unique->ExogenousDependencies();


            std::unordered_set<std::string> three_addr_seen;

            auto IB = std::make_shared<InstructionBlock>();

            auto deps = unique->DepthFirstAnySymbolicDependencyAndThis();
            for (auto head : deps.DepthFirst) {
                if (head->IsExo())
                    continue;
                if (three_addr_seen.count(head->Name()) == 0) {
                    auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                    IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                    three_addr_seen.insert(head->Name());
                }

            }
            IB->Add(std::make_shared< InstructionReturn>(deps.DepthFirst.back()->Name()));


            auto AADIB = std::make_shared<InstructionBlock>();

            // IB->EmitCode(std::cout);

            std::vector< std::shared_ptr<InstructionDeclareVariable> > decl_list;

            std::vector<std::shared_ptr< ImpliedMatrixFunction >> matrix_func_list;

            std::unordered_map<std::string, size_t> head_alloc_map;
            for (size_t idx = 0; idx != arguments.size(); ++idx)
            {
                size_t slot = head_alloc_map.size();
                head_alloc_map[arguments[idx]] = slot;
            }
            std::vector< std::unordered_map<std::string, size_t> > allocation_map_list{ head_alloc_map };




            for (auto const& instr : *IB)
            {



                if (auto decl_instr = std::dynamic_pointer_cast<InstructionDeclareVariable>(instr))
                {

                    auto alloc_map = allocation_map_list.back();

                    decl_list.push_back(decl_instr);

                    std::vector<std::string> comment_vec;

                    auto expr = decl_instr->as_operator_();
                    auto matrix_func = ImpliedMatrixFunction::Make(matrix_func_list.size(), instr);

                    size_t slot = alloc_map.size();
                    alloc_map[decl_instr->LValueName()] = slot;

                    AADIB->Add(matrix_func->MakeComment());
                    AADIB->Add(decl_instr);

                    matrix_func_list.push_back(matrix_func);
                    allocation_map_list.push_back(alloc_map);


                }
                else
                {
                    // default
                    AADIB->Add(instr);
                }

            }

            AADIB->Add(std::make_shared<InstructionComment>(std::vector<std::string>{ "//////////////", "Starting AAD matrix", "//////////////", }));
            std::vector<std::string> matrix_lvalues;

            std::vector<std::shared_ptr<SymbolicMatrix> > adj_matrix_list;
            for (size_t idx = matrix_func_list.size(); idx != 0; )
            {
                bool is_terminal = (idx == matrix_func_list.size());
                --idx;
                auto const& alloc_map = allocation_map_list[idx];
                //AADIB->Add(matrix_func_list[idx]->MakeComment());

                auto matrix_decl = matrix_func_list[idx]->MakeMatrix(alloc_map, is_terminal);

                adj_matrix_list.push_back(matrix_decl->Matrix());

                matrix_lvalues.push_back(matrix_decl->LValueName());
                //AADIB->Add(matrix_decl);
            }

            // fold matrix list
            std::shared_ptr<SymbolicMatrix> adj_matrix = adj_matrix_list[0];
            for (size_t idx = 1; idx < adj_matrix_list.size(); ++idx)
            {
                adj_matrix = adj_matrix_list[idx]->Multiply(*adj_matrix);
            }

            // ADIB->Add(std::make_shared < InstructionDeclareMatrix> ("DD", adj_matrix));

#if 0
            for (size_t idx = 0; idx != arguments.size(); ++idx)
            {
                auto d_sym = std::string("d_") + arguments[idx];
                auto d_expr = adj_matrix->At(idx, 0);
                std::stringstream ss;
                ss << "if( " << d_sym << ") { *" << d_sym << " = ";
                d_expr->EmitCode(ss);
                ss << "; }";
                AADIB->Add(std::make_shared< InstructionText>(ss.str()));
            }
#endif

            for (size_t idx = 0; idx != arguments.size(); ++idx)
            {
                auto d_sym = std::string("d_") + arguments[idx];
                auto d_expr_orig = adj_matrix->At(idx, 0);
                auto d_expr = d_expr_orig->Clone(RU);


                std::stringstream ss;
#if 0
                ss << "const double debug__" << d_sym << " = ";
                d_expr_orig->EmitCode(ss);
                ss << ";";
                AADIB->Add(std::make_shared< InstructionText>(ss.str()));
#endif

                auto deps = d_expr->DepthFirstAnySymbolicDependencyAndThis();
                for (auto head : deps.DepthFirst) {
                    if (head->IsExo())
                        continue;
                    if (three_addr_seen.count(head->Name()) == 0) {
                        auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                        AADIB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                        three_addr_seen.insert(head->Name());
                    }

                }

                std::string aux_name = "result_" + d_sym;
                AADIB->Add(std::make_shared<InstructionDeclareVariable>(aux_name, d_expr));

                ss.str("");
                ss << "if( " << d_sym << ") { *" << d_sym << " = " << aux_name << "; }";
                AADIB->Add(std::make_shared< InstructionText>(ss.str()));
            }

            return AADIB;
        }

        void EmitToString(std::ostream& ostr)const
        {
            auto ad_kernel = Function::template Build<DoubleKernel>();

            auto AADIB = this->GenerateInstructionBlock();

            auto arguments = ad_kernel.Arguments();

            std::vector<std::string> arg_list;
            for (size_t idx = 0; idx != arguments.size(); ++idx)
            {
                arg_list.push_back("double " + arguments[idx]);
            }
            for (size_t idx = 0; idx != arguments.size(); ++idx)
            {
                arg_list.push_back("double* d_" + arguments[idx] + " = nullptr");
            }


            ostr << "double " << ad_kernel.Name() << "(";
            for (size_t idx = 0; idx != arg_list.size(); ++idx)
            {
                ostr << (idx == 0 ? "" : ", ") << arg_list[idx];
            }
            ostr << ")\n{\n";
            for (auto const& instr : *AADIB)
            {
                instr->EmitCode(ostr);
                ostr << "\n";
            }
            ostr << "}\n";
        }
        std::string GenerateString()const
        {
            std::stringstream ss;
            this->EmitToString(ss);
            return ss.str();
        }

    };


} // end namespace Cady

#endif // INCLUDE_ADD_FUNCTION_GENERATOR_H