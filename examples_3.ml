(* the rest is applications of this *)
let simplest_graph = (Dist (Variable "bernoulli", Bernoulli (Constant 0.5)));;
let normal_graph = (Dist (Variable "x", 
	(Normal ((Dist (Variable "latent_normal", Normal( Dist (Variable "bernoulli", Bernoulli (Constant 0.5)),
					          (Constant 10.)))),
		(Constant 1.)))));;
let normal_subgraph = Dist (Variable "x", Normal( Dist (Variable "bernoulli", Bernoulli (Constant 0.5)), (Constant 1.)));;

let observed = Hashtbl.Poly.of_alist_exn [(Variable "x", Torch.Tensor.float_vec [6.; 6.2; 6.3; 6.2; 6.1; 11.0; 11.1; 11.2; 11.1; 11.5; 10.8;])];;			 

 get_sample_from_graph normal_graph;;

let sampling_map = Map.Poly.of_alist_exn ((get_sample_from_graph observed normal_graph) |> snd3);;

let some_result = inference_by_naive_sampling observed normal_graph 1000;;
variable_hist some_result "latent_normal";;


log_likelihood_normal (Tensor.float_vec [6.; 6.5; 6.5; 7.; 7.; 6.;])  0. 2.;;
log_likelihood_bernoulli (Tensor.float_vec [0.; 0.; 0.; 1.; 1.; 0.; 0.;]) 0.2;;

get_all_summary_statistics some_result;;
(* - : (variable * (string * float) list) list =
[(Variable "bernoulli",
  [("mean", 0.428847884964032089); ("variance", 1.0544988204073866e+20)]);
 (Variable "latent_normal",
  [("mean", 8.83415662206087227); ("variance", 7.51065961291537197e+22)]);
 (Variable "x",
  [("mean", 6.00000000000000533); ("variance", 1.1744195081213079e-08)])]
*)
