(* Examples 1 *)


log_likelihood_normal (Tensor.float_vec [6.; 6.5; 6.5; 7.; 7.; 6.;])  0. 2.;;
log_likelihood_bernoulli (Tensor.float_vec [0.; 0.; 0.; 1.; 1.; 0.; 0.;]) 0.2;;


let simplest_graph = (Dist (Variable "bernoulli", Bernoulli (Constant 0.5)));;
(* let sum_graph_regression =  *)
let normal_subgraph = Dist (Variable "x", Normal( Dist (Variable "bernoulli", Bernoulli (Constant 0.5)), (Constant 1.)));;

(* Example 2 *)
let normal_graph = (Dist (Variable "x", 
	(Normal ((Dist (Variable "latent_normal", Normal( Dist (Variable "bernoulli", Bernoulli (Constant 0.5)),
					          (Constant 10.)))),
		(Constant 1.)))));;
		
let observed = Hashtbl.Poly.of_alist_exn [(Variable "x", Torch.Tensor.float_vec [6.; 6.2; 6.3; 6.2; 6.1; 11.0; 11.1; 11.2; 11.1; 11.5; 10.8;])];;			 
get_sample_from_graph observed normal_graph 0;;

let sampling_map = Map.Poly.of_alist_exn ((get_sample_from_graph observed normal_graph 0) |> snd3);;

let some_result = inference_by_naive_sampling observed normal_graph 1000;;
variable_hist some_result "latent_normal";;


get_all_summary_statistics some_result;;
(* - : (variable * (string * float) list) list =
[(Variable "bernoulli",
  [("mean", 0.428847884964032089); ("variance", 1.0544988204073866e+20)]);
 (Variable "latent_normal",
  [("mean", 8.83415662206087227); ("variance", 7.51065961291537197e+22)]);
 (Variable "x",
  [("mean", 6.00000000000000533); ("variance", 1.1744195081213079e-08)])]
*)


(* Example 3: Linear regression! *)
(* Model graph *)
let var_x = Dist (Variable "x", Normal( Constant 1., Constant 10.));;
let var_w = Dist (Variable "w", Normal( Constant 1., Constant 2.));;
let var_y = Dist (Variable "y", Normal( Binop (Variable "deterministic_y_mean", Multiply, var_x, var_w),
		  			Constant 10.));;
let linear_regression_graph = var_y;;

(* Data *)
let observed_values_linear_regression = Hashtbl.Poly.of_alist_exn [(Variable "x", Torch.Tensor.float_vec [0.; 0.1; 0.5; 0.8; 1.; 1.5; 2.; 3.; 3.5; 4.; 4.5; 5.0; 5.5; 6.0; 6.5; 7.0;]);
			  		                           (Variable "y", Torch.Tensor.float_vec [0.; 0.2; 1.0; 1.6; 2.; 3.0; 4.; 6.; 7.0; 8.; 9.0; 10.; 11.; 12.; 13.; 14.0;])];;							    
			  		                           
(* Inference *)
let linear_regression_samples = inference_by_naive_sampling observed_values_linear_regression linear_regression_graph 10000;;
get_all_summary_statistics linear_regression_samples;;

(* Similar result with 12x the samples *)
let linear_regression_samples = inference_by_naive_sampling observed_values_linear_regression linear_regression_graph 120000;;
get_all_summary_statistics linear_regression_samples;;

