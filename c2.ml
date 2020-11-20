(* ************************************************************************ *)
(* ************************************************************************ *)

(* PPL 2: Consists of 
1) constants
2) random variables (bernoulli or normal or N(0,1))
3) observed data 

Shape of everything is scalar.
No additions yet!

Sample and Observe is the same statement now.
*)

(* ************************************************************************ *)
type variable = Variable of string;;

type op1 = Sample;;
type op2 = Add | Multiply;;
type dist = Normal of expr * expr | StandardNormal | Bernoulli of expr and
     expr = Dist of variable * dist | Unop of variable * op1 * expr | Constant of float;;

(* ************************************************************************ *) 
open Core
open Torch

let sample_standard_normal = Tensor.get_float1 (Tensor.normal_ ~mean:0. ~std:1. (Tensor.ones [1])) 0 ;;
let sample_normal ~mu ~sigma = Tensor.get_float1 (Tensor.normal_ ~mean:mu ~std:sigma (Tensor.ones [1])) 0 ;;
let sample_bernoulli ~p = if (Tensor.get_float1 (Tensor.rand [1]) 0 |> Float.compare p) = -1 then 0. else 1. ;;
(* I looked at https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood *)	
let log_likelihood_normal xs mu var = 
	let n = Float.of_int (Tensor.shape1_exn xs) in
	let s1 = -. n /. 2. *. Float.log (2. *. Float.pi) in
	let s2 = -. n /. 2. *. Float.log(var) in
	let s3 = Tensor.to_float0_exn (Tensor.sum (Tensor.square (Tensor.sub1 xs (Scalar.f mu)))) *.
			     (-1. /. (2. *. var)) in
	s1 +. s2 +. s3
	
let log_likelihood_standard_normal xs = log_likelihood_normal xs 0. 1.
(* Took this from https://web.stanford.edu/class/cs109/reader/11%20Parameter%20Estimation.pdf *)
let log_likelihood_bernoulli xs p = 
  let n = Float.of_int (Tensor.shape1_exn xs) in
  let y = Tensor.sum xs |> Tensor.to_float0_exn in 
  y *. (Float.log p) +. (n -. y) *. Float.log (1. -. p)

(* Returns a tuple of: 1 value from the leaf, and some sort of map for the sampled values for every variable *)
let rec get_sample_from_graph_prob observed_map obs_idx g = 
    match g with 
    | Dist (v, Bernoulli p_expr) -> sample_bernoulli ~p:(get_sample_from_graph_prob observed_map obs_idx p_expr)
    | Dist (v, StandardNormal) -> sample_standard_normal
    | Dist (v, Normal(mu_expr, sigma_expr)) -> sample_normal ~mu:(get_sample_from_graph_prob observed_map obs_idx mu_expr) ~sigma:(get_sample_from_graph_prob observed_map obs_idx sigma_expr)
    | Constant (c) -> c
    | Unop (v, Sample, e) -> get_sample_from_graph_prob observed_map obs_idx e 
    | Unop (v, Observe, e) -> Tensor.get_float1 (Hashtbl.Poly.find_exn observed_map v) obs_idx
    ;;


(* let obs = Tensor.get_float1 (Hashtbl.Poly.find_exn observed_map v) idx in *)
(* ENH: Just get the latest values directly from a stack? *)
let rec get_sample_from_graph g = 
    match g with 
    | Dist (v, Bernoulli p_expr) -> 
    	let p, alist, w = (get_sample_from_graph p_expr) in
    	let bernoulli_sample = sample_bernoulli ~p in
    	bernoulli_sample, alist @ [(v, bernoulli_sample)], 0
    | Dist (v, StandardNormal) ->
    	let sn_sample = sample_standard_normal in
    	sn_sample, [(v, sn_sample)], 0
    | Dist (v, Normal(mu_expr, sigma_expr)) -> 
    	let mu, mu_alist, w_mu = get_sample_from_graph mu_expr in
    	let sigma, sigma_alist, w_sigma = get_sample_from_graph sigma_expr in
    	let n_sample = sample_normal ~mu:(mu) ~sigma:(sigma) in
    	n_sample, mu_alist @ sigma_alist @ [(v, n_sample)], 0
    | Constant (c) -> c, [], 0
    | Unop (v, Sample, e) -> 
    	let sample_v, sample_alist, w = get_sample_from_graph e in
    	sample_v, sample_alist @ [(v, sample_v)], 0
    ;;

(* Return a list of log-likelihoods: 1 for every Observe statement. *)
let rec get_loglikelihoods_from_graph sampling_map observed_map g =
    match g with 
    | Dist (v, Normal(e1, e2)) -> [] @ (get_loglikelihoods_from_graph sampling_map observed_map e1) @ (get_loglikelihoods_from_graph sampling_map observed_map e2)
    | Dist (v, StandardNormal) -> []
    | Dist (v, Bernoulli e) -> (get_loglikelihoods_from_graph sampling_map observed_map e)
    (* Advance 1, doesn't matter *)
    | Constant (c) -> []
    | Unop (v, Sample, e) ->
        (* Advance 1, the next one may be it *)
    	get_loglikelihoods_from_graph sampling_map observed_map e
    | Unop (v, Observe, Unop (_, Sample, (Dist(v2, d))))
    | Unop (v, Observe, Dist (v2, d)) ->
    	let observed_vs = (Hashtbl.Poly.find_exn observed_map v) in
    	match d with 
    	| StandardNormal -> [(log_likelihood_standard_normal observed_vs)]
	| Bernoulli (Constant c) -> [(log_likelihood_bernoulli observed_vs c)]
	| Bernoulli (Dist (v3, _)) -> [(log_likelihood_bernoulli observed_vs (Map.Poly.find_exn sampling_map v3))] (* @ (get_loglikelihoods_from_graph sampling_map observed_map e) *)
	| Normal (Constant c1, Constant c2) -> [(log_likelihood_normal observed_vs c1 c2)]
	| Normal (Constant c1, (Dist (v32, _))) -> [log_likelihood_normal observed_vs c1 (Map.Poly.find_exn sampling_map v32)] (* @ (get_loglikelihoods_from_graph sampling_map observed_map e2) *)
	| Normal ((Dist (v31, _)), Constant c2) -> [log_likelihood_normal observed_vs (Map.Poly.find_exn sampling_map v31) c2] (* @ (get_loglikelihoods_from_graph sampling_map observed_map e1) *)
	| Normal ((Dist (v31, _)), Dist (v32, _)) -> [(log_likelihood_normal observed_vs (Map.Poly.find_exn sampling_map v31) (Map.Poly.find_exn sampling_map v32))] (* @ (get_loglikelihoods_from_graph sampling_map observed_map e1) @ (get_loglikelihoods_from_graph sampling_map observed_map e2) *)
;;        

let inference_by_naive_sampling observed_map graph n_iters =
    let rec iter iter_number = 
        let samples = get_sample_from_graph graph |> snd in
        let loglikelihood = get_loglikelihoods_from_graph (samples |> Map.Poly.of_alist_exn) observed_map graph |>
                            List.fold ~init:1. ~f:(fun acc x -> acc *. x) in
        let sampling_alist =  List.map samples ~f:(fun (var_name, sample) -> (var_name, (sample, loglikelihood))) in
        if (iter_number >= n_iters) then sampling_alist else sampling_alist @ (iter (iter_number + 1)) in
   Map.Poly.of_alist_multi (iter 0);;


(* TODO: Wrong, until matplotlib dist can plot with weights... *)
let variable_hist result_map x = 
  Pyplot.hist ((List.map ~f:fst (Map.Poly.find_exn result_map (Variable x))) |> Array.of_list);
  Mpl.show()

