(* the rest is applications of this *)
let simplest_graph = Unop (Variable "s", Sample , Dist (Variable "bernoulli", Bernoulli (Constant 0.5)));;
let normal_graph = Unop (Variable "x",
			 Sample,
			 Dist (Variable "normal", Normal( Dist (Variable "bernoulli", Bernoulli (Constant 0.5)), (Constant 1.))));;			 
let normal_subgraph = Dist (Variable "normal", Normal( Dist (Variable "bernoulli", Bernoulli (Constant 0.5)), (Constant 1.)));;
let observed = Hashtbl.Poly.of_alist_exn [(Variable "x", Torch.Tensor.float_vec [6.; 6.2; 6.3; 6.2; 6.1; 11.0; 11.1; 11.2; 11.1; 11.5; 10.8;])];;			 

 get_sample_from_graph normal_graph;;
(* - : float * (variable * float) list =
(6.,
 [(Variable "bernoulli", 1.); (Variable "normal", 2.2070167064666748);
  (Variable "x", 6.)])
*)

let sampling_map = Map.Poly.of_alist_exn ((get_sample_from_graph normal_graph) |> snd);;

get_loglikelihoods_from_graph sampling_map observed normal_graph;;

(*
- : float list = [-475.873338513688907]
*)

let some_result = inference_by_naive_sampling observed normal_subgraph 1000;;
variable_hist some_result "normal";;


log_likelihood_normal (Tensor.float_vec [6.; 6.5; 6.5; 7.; 7.; 6.;])  0. 2.;;
log_likelihood_bernoulli (Tensor.float_vec [0.; 0.; 0.; 1.; 1.; 0.; 0.;]) 0.2;;
