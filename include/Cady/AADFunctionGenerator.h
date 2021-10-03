#ifndef INCLUDE_ADD_FUNCTION_GENERATOR_H
#define INCLUDE_ADD_FUNCTION_GENERATOR_H

#include "Cady.h"
#include "CpuTimer.h"

namespace Cady {


    class AADFunctionGeneratorPersonality
    {
    public:
        bool matrix_function_comments{ true };
    };

    class FunctionGenerator
    {
    public:
        virtual ~FunctionGenerator() = default;
        virtual std::shared_ptr<Function> GenerateInstructionBlock(AADFunctionGeneratorPersonality personality = AADFunctionGeneratorPersonality{})const = 0;
    };

    template<class Kernel>
    class SimpleFunctionGenerator : public FunctionGenerator
    {
    public:

        std::shared_ptr<Module> BuildRecursive(std::shared_ptr<Operator> const& head)const
        {
            struct IfFinder
            {
                void operator()(std::shared_ptr<Operator> const& op)
                {
                    if (auto if_stmt = std::dynamic_pointer_cast<If>(op))
                    {
                        if_stmts_.push_back(if_stmt);
                    }
                }

                std::vector<std::shared_ptr<If> > if_stmts_;
            };

            auto if_finder = std::make_shared< IfFinder>();
            head->VisitTopDown(*if_finder);

            if (if_finder->if_stmts_.size() > 0)
            {
                auto if_expr = if_finder->if_stmts_[0];

                // I expect this to be a block of variable declerations,
                // the last decleration will be the conditional variable
                auto cond = BuildRecursive(if_expr->Cond());
                auto cond_module = std::dynamic_pointer_cast<Module>(cond);
                auto cond_ib = std::dynamic_pointer_cast<InstructionBlock>(cond_module->back());

                auto cond_var_name = [&]()->std::string
                {
                    if (auto last_instr = std::dynamic_pointer_cast<InstructionReturn>(cond_ib->back()))
                    {
                        auto var_name = last_instr->VarName();
                        cond_ib->pop_back();
                        return var_name;
                    }
                    throw std::runtime_error("unexpected");
                }();

                auto if_true = BuildRecursive(if_expr->IfTrue());
                auto if_false = BuildRecursive(if_expr->IfFalse());

                auto if_block = std::make_shared< IfBlock>(
                    cond_var_name,
                    if_true,
                    if_false);

                auto modulee = std::make_shared<Module>();
                modulee->push_back(cond_ib);
                modulee->push_back(if_block);
                return modulee;
            }
            else
            {
                std::unordered_set<std::string> symbols_seen;

                auto IB = std::make_shared<InstructionBlock>();

                auto deps = head->DepthFirstAnySymbolicDependencyAndThis();
                if (deps.DepthFirst.size() > 0)
                {
                    for (auto sym : deps.DepthFirst) {
                        if (sym->IsExo())
                            continue;
                        if (symbols_seen.count(sym->Name()) != 0)
                        {
                            continue;
                        }
                        auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(sym)->Expr();
                        IB->Add(std::make_shared<InstructionDeclareVariable>(sym->Name(), expr));
                        symbols_seen.insert(sym->Name());

                    }
                    IB->Add(std::make_shared< InstructionReturn>(deps.DepthFirst.back()->Name()));
                }
                else
                {
                    if (auto symbol = std::dynamic_pointer_cast<Symbol>(head))
                    {
                        IB->Add(std::make_shared< InstructionReturn>(symbol->Name()));
                    }
                    else
                    {
                        throw std::domain_error("should not be possible");
                    }
                }
                

                auto modulee = std::make_shared<Module>();
                modulee->push_back(IB);
                return modulee;

            }
        }

        std::shared_ptr<Function> GenerateInstructionBlock(AADFunctionGeneratorPersonality personality = AADFunctionGeneratorPersonality{})const
        {
            // First we evaulate the function, in order to get an expression treee
            auto ad_kernel = Kernel::template Build<DoubleKernel>();

            auto arguments = ad_kernel.Arguments();

            std::vector<DoubleKernel> symbolc_arguments;
            for (auto const& arg : arguments)
            {
                symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
            }
            auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

            auto expr = function_root.as_operator_();

            auto head_control_block = BuildRecursive(expr);



            auto f = std::make_shared<Function>(head_control_block);
            for (auto const& arg : arguments)
            {
                f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
            }

            return f;
        };
    };




    template<class Kernel>
    class SingleExprFunctionGenerator : public FunctionGenerator
    {
    public:
        std::shared_ptr<Function> GenerateInstructionBlock(AADFunctionGeneratorPersonality personality = AADFunctionGeneratorPersonality{})const
        {
            // First we evaulate the function, in order to get an expression treee
            auto ad_kernel = Kernel::template Build<DoubleKernel>();

            auto arguments = ad_kernel.Arguments();

            std::vector<DoubleKernel> symbolc_arguments;
            for (auto const& arg : arguments)
            {
                symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
            }
            auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

            struct RemoveEndo : OperatorTransform {
                virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {
                    auto candidate = ptr->Clone(shared_from_this());
                    if (candidate->Kind() == OPKind_EndgenousSymbol) {
                        if (auto typed = std::dynamic_pointer_cast<EndgenousSymbol>(candidate)) {
                            return typed->Expr();
                        }
                    }
                    return candidate;
                }
            };

            auto expr = function_root.as_operator_()->Clone(std::make_shared< RemoveEndo>());

            // maintain information about what symbol I've emitted
            std::unordered_set<std::string> symbols_seen;

            auto IB = std::make_shared<InstructionBlock>();



            auto deps = expr->DepthFirstAnySymbolicDependencyAndThis();
            for (auto head : deps.DepthFirst) {
                if (head->IsExo())
                    continue;
                if (symbols_seen.count(head->Name()) != 0)
                {
                    continue;
                }
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                symbols_seen.insert(head->Name());

            }
            IB->Add(std::make_shared< InstructionReturn>(deps.DepthFirst.back()->Name()));

            auto f = std::make_shared<Function>(IB);
            for (auto const& arg : arguments)
            {
                f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
            }

            return f;
        };
    };


    template<class Kernel>
    class ThreeAddressFunctionGenerator : public FunctionGenerator
    {
    public:
        std::shared_ptr<Function> GenerateInstructionBlock(AADFunctionGeneratorPersonality personality = AADFunctionGeneratorPersonality{})const
        {
            // First we evaulate the function, in order to get an expression treee
            auto ad_kernel = Kernel::template Build<DoubleKernel>();

            auto arguments = ad_kernel.Arguments();

            std::vector<DoubleKernel> symbolc_arguments;
            for (auto const& arg : arguments)
            {
                symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
            }
            auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

            auto expr = function_root.as_operator_();


            // now we transform the expression into threee address code
            auto three_address_transform = std::make_shared<Transform::RemapUnique>();
            auto three_address_tree = expr->Clone(three_address_transform);

            // maintain information about what symbol I've emitted
            std::unordered_set<std::string> three_addr_seen;

            auto IB = std::make_shared<InstructionBlock>();



            auto deps = three_address_tree->DepthFirstAnySymbolicDependencyAndThis();
            for (auto head : deps.DepthFirst) {
                if (head->IsExo())
                    continue;
                if (three_addr_seen.count(head->Name()) != 0)
                {
                    continue;
                }
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                three_addr_seen.insert(head->Name());

            }
            IB->Add(std::make_shared< InstructionReturn>(deps.DepthFirst.back()->Name()));



            auto f = std::make_shared<Function>(IB);
            for (auto const& arg : arguments)
            {
                f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
            }

            return f;
        };
    };


    template<class Kernel>
    class ForwardDiffFunctionGenerator : public FunctionGenerator
    {
    public:
        std::shared_ptr<Function> GenerateInstructionBlock(AADFunctionGeneratorPersonality personality = AADFunctionGeneratorPersonality{})const
        {
            // First we evaulate the function, in order to get an expression treee
            auto ad_kernel = Kernel::template Build<DoubleKernel>();

            auto arguments = ad_kernel.Arguments();

            std::vector<DoubleKernel> symbolc_arguments;
            for (auto const& arg : arguments)
            {
                symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
            }
            auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

            struct RemoveEndo : OperatorTransform {
                virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {
                    auto candidate = ptr->Clone(shared_from_this());
                    if (candidate->Kind() == OPKind_EndgenousSymbol) {
                        if (auto typed = std::dynamic_pointer_cast<EndgenousSymbol>(candidate)) {
                            return typed->Expr();
                        }
                    }
                    return candidate;
                }
            };

            auto expr = [&]()
            {
                auto ptr = function_root.as_operator_()->Clone(std::make_shared< RemoveEndo>());
                if (auto sym = std::dynamic_pointer_cast<EndgenousSymbol>(ptr))
                {
                    return sym->Expr();
                }
                return ptr;
            }();

            std::vector<std::shared_ptr<Operator> > diff_list;
            for (auto const& arg : arguments)
            {
                diff_list.push_back(expr->Diff(arg));
            }

            // maintain information about what symbol I've emitted
            std::unordered_set<std::string> symbols_seen;

            auto IB = std::make_shared<InstructionBlock>();


            std::vector<std::shared_ptr<Operator> > all_exprs{ expr };
            std::copy(diff_list.begin(), diff_list.end(), std::back_inserter(all_exprs));

            std::vector<std::shared_ptr<Operator> > result_list;

            auto three_address_transform = std::make_shared<Transform::RemapUnique>();

            for (auto const& child_expr : all_exprs)
            {
                auto three_addr_child_expr = child_expr->Clone(three_address_transform);
                auto deps = three_addr_child_expr->DepthFirstAnySymbolicDependencyAndThis();
                for (auto head : deps.DepthFirst) {
                    if (head->IsExo())
                        continue;
                    if (symbols_seen.count(head->Name()) != 0)
                    {
                        continue;
                    }
                    auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                    IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                    symbols_seen.insert(head->Name());

                }
                result_list.push_back(deps.DepthFirst.back());
            }

            for (size_t arg_index = 0; arg_index != arguments.size(); ++arg_index)
            {
                auto const& arg = arguments[arg_index];
                std::string d_sym = "d_" + arg;
                std::string aux_name = "result_" + d_sym;

                auto result_sym = result_list[arg_index + 1];



                
                IB->Add(std::make_shared<InstructionDeclareVariable>(aux_name, result_sym));

                std::stringstream ss;
                ss << "if( " << d_sym << ") { *" << d_sym << " = " << aux_name << "; }";
                IB->Add(std::make_shared< InstructionText>(ss.str()));

            }

            std::string result_name{ "result" };
            IB->Add(std::make_shared<InstructionDeclareVariable>(result_name,result_list[0]));
            IB->Add(std::make_shared< InstructionReturn>(result_name));

            auto f = std::make_shared<Function>(IB);
            f->SetFunctionName(ad_kernel.Name());
            for (auto const& arg : arguments)
            {
                f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
            }
            for (auto const& arg : arguments)
            {
                f->AddArg(std::make_shared<FunctionArgument>(FAK_OptDoublePtr, "d_" + arg));
            }

            return f;
        };
    };



    template<class Kernel>
    class AADFunctionGenerator : public FunctionGenerator
    {
    public:
        enum AADPropogationType
        {
            AADPT_Forwards,
            AADPT_Backwards,
        };
    
        
        AADFunctionGenerator(AADPropogationType propogation_type)
        {
            propogation_type_ = propogation_type;
        }
        std::shared_ptr<Function> GenerateInstructionBlock(AADFunctionGeneratorPersonality personality = AADFunctionGeneratorPersonality{})const
        {
            // First we evaulate the function, in order to get an expression treee
            auto ad_kernel = Kernel::template Build<DoubleKernel>();

            auto arguments = ad_kernel.Arguments();

            std::vector<DoubleKernel> symbolc_arguments;
            for (auto const& arg : arguments)
            {
                symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
            }
            auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

            auto expr = function_root.as_operator_();


            // now we transform the expression into threee address code
            auto three_address_transform = std::make_shared<Transform::RemapUnique>();
            auto three_address_tree = expr->Clone(three_address_transform);

            // maintain information about what symbol I've emitted
            std::unordered_set<std::string> three_addr_seen;

            auto IB = std::make_shared<InstructionBlock>();

            auto deps = three_address_tree->DepthFirstAnySymbolicDependencyAndThis();
            for (auto head : deps.DepthFirst) {
                if (head->IsExo())
                    continue;
                if (three_addr_seen.count(head->Name()) != 0)
                {
                    continue;
                }
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                three_addr_seen.insert(head->Name());

            }
            IB->Add(std::make_shared< InstructionReturn>(deps.DepthFirst.back()->Name()));


            // now we have an instruction block IB, which evaluate the function

            auto AADIB = std::make_shared<InstructionBlock>();

            // There I want to transform each three address code into an implied function
            std::vector<std::shared_ptr< ImpliedMatrixFunction >> matrix_func_list;

            // need to maintain an mapping of symbol to matrix index
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
                    auto expr = decl_instr->as_operator_();
                    auto matrix_func = ImpliedMatrixFunction::Make(matrix_func_list.size(), instr);

                    auto alloc_map = allocation_map_list.back();
                    size_t slot = alloc_map.size();
                    alloc_map[decl_instr->LValueName()] = slot;

                    AADIB->Add(matrix_func->MakeComment());
                    AADIB->Add(decl_instr);

                    // store this for later, in order to compute the derivatives
                    matrix_func_list.push_back(matrix_func);
                    allocation_map_list.push_back(alloc_map);


                }
                else
                {
                    // default
                    AADIB->Add(instr);
                }

            }

            AADIB->Add(std::make_shared<InstructionComment>(std::vector<std::string>{ "// AD section", }));

            // collect matrix list
            std::vector<std::shared_ptr<SymbolicMatrix> > adj_matrix_list;
            for (size_t idx = matrix_func_list.size(); idx != 0; )
            {
                bool is_terminal = (idx == matrix_func_list.size());
                --idx;
                auto const& alloc_map = allocation_map_list[idx];

                auto matrix_decl = matrix_func_list[idx]->MakeMatrix(alloc_map, is_terminal);

                adj_matrix_list.push_back(matrix_decl->Matrix());

            }


            // fold matrix list in reverse

            auto fold_backwards = [&]()
            {
                // (M1*(M2*M3))
                std::shared_ptr<SymbolicMatrix> adj_matrix = adj_matrix_list[0];
                for (size_t idx = 1; idx < adj_matrix_list.size(); ++idx)
                {
                    adj_matrix = adj_matrix_list[idx]->Multiply(*adj_matrix);
                }
                return adj_matrix;
            };

            auto fold_forwards = [&]()
            {
                auto rev_order = std::vector<std::shared_ptr<SymbolicMatrix> >(adj_matrix_list.rbegin(), adj_matrix_list.rend());

                // (M1*M2)*M3)
                std::shared_ptr<SymbolicMatrix> adj_matrix = rev_order[0];
                for (size_t idx =1; idx <rev_order.size();++idx)
                {
                    adj_matrix = adj_matrix->Multiply(*rev_order[idx]);
                }
                
                return adj_matrix;
            };

            auto adj_matrix = [&]()-> std::shared_ptr<SymbolicMatrix>
            {
                if (propogation_type_ == AADPT_Backwards)
                {
                    return fold_backwards();
                }
                else if (propogation_type_ == AADPT_Forwards)
                {
                    return fold_forwards();
                }
                else
                {
                    return {};
                }
            }();



            // emit derivies
            for (size_t idx = 0; idx != arguments.size(); ++idx)
            {
                auto d_sym = std::string("d_") + arguments[idx];
                auto d_expr_orig = adj_matrix->At(idx, 0);
                Transform::FoldZero fold_zero;
                auto d_expr_no_zero = fold_zero.Fold(d_expr_orig);
                auto d_expr_three_address = d_expr_no_zero->Clone(three_address_transform);


                

                auto deps = d_expr_three_address->DepthFirstAnySymbolicDependencyAndThis();
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
                AADIB->Add(std::make_shared<InstructionDeclareVariable>(aux_name, d_expr_three_address));

                std::stringstream ss;
                ss << "if( " << d_sym << ") { *" << d_sym << " = " << aux_name << "; }";
                AADIB->Add(std::make_shared< InstructionText>(ss.str()));
            }

            auto f = std::make_shared<Function>(AADIB);
            f->SetFunctionName(ad_kernel.Name());
            for (auto const& arg : arguments)
            {
                f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
            }
            for (auto const& arg : arguments)
            {
                f->AddArg(std::make_shared<FunctionArgument>(FAK_OptDoublePtr, "d_" + arg));
            }

            return f;
        }
    private:
        AADPropogationType propogation_type_;
    };


} // end namespace Cady

#endif // INCLUDE_ADD_FUNCTION_GENERATOR_H