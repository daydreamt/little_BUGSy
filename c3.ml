(* ************************************************************************ *)
(* ************************************************************************ *)

(* PPL 2: Consists of 
1) constants
2) random variables (bernoulli or normal or N(0,1))
3) observed data 

Shape of everything is scalar.
No additions yet!

Sample and Observe is the same statement now. But there is no need for it, in the graph, since we can condition on any variable, so we remove it, further simplifying our code!

*)

(* ************************************************************************ *)

(* | Unop of variable * op1 * expr  *)

type variable = Variable of string;;

type op1 = Sample | Observe;;
type op2 = Add | Multiply;;
type dist = Normal of expr * expr | StandardNormal | Bernoulli of expr and
     expr = Dist of variable * dist | Constant of float;;

(* ************************************************************************ *) 
#require "torch.toplevel";;
#require "core";;
#require "matplotlib";;

open Core
open Torch
open Matplotlib

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


let get_first_float_of_tensor tensor = Tensor.get_float1 tensor 0;;

(* let obs = Tensor.get_float1 (Hashtbl.Poly.find_exn observed_map v) idx in *)
(* ENH: Just get the latest values directly from a stack? *)
(* FIXME: Just like the previous versions, doesn't work correctly with non-leaves observed
 values, as they would then have to propagate a tensor, and make the sample 
 functions work with different sample shapes. This still needs some thought.*)
let rec get_sample_from_graph observed g = 
    match g with 
    | Dist (v, Bernoulli p_expr) ->
    	let p, alist, w = (get_sample_from_graph observed p_expr) in
    	begin
    	match Hashtbl.find observed v with
    	| None ->
	    let bernoulli_sample = sample_bernoulli ~p in
	    bernoulli_sample, alist @ [(v, bernoulli_sample)], w
	| Some s ->
	    (Tensor.get_float1 s 0), alist @ [(v, (Tensor.get_float1 s 0))], (log_likelihood_bernoulli s p) +. w
	end
    | Dist (v, StandardNormal) ->
    	let sn_sample = sample_standard_normal in
    	sn_sample, [(v, sn_sample)], 0.
    | Dist (v, Normal(mu_expr, sigma_expr)) ->
    	let mu, mu_alist, w_mu = get_sample_from_graph observed mu_expr in
        let var, var_alist, w_sigma = get_sample_from_graph observed sigma_expr in
        begin
    	match Hashtbl.find observed v with
    	| None ->
    	    let n_sample = sample_normal ~mu:(mu) ~sigma:(var) in
    	    n_sample, mu_alist @ var_alist @ [(v, n_sample)], 0.
        | Some s ->
            (Tensor.get_float1 s 0), mu_alist @ var_alist @ [(v, (Tensor.get_float1 s 0))], (log_likelihood_normal s mu var) +. w_mu +. w_sigma
        end
    | Constant (c) -> c, [], 0.
    ;;


let inference_by_naive_sampling observed_map graph n_iters =
    let rec iter iter_number = 
        let _, samples, loglikelihood = get_sample_from_graph observed_map graph in
        let sampling_alist =  List.map samples ~f:(fun (var_name, sample) -> (var_name, (sample, loglikelihood))) in
        if (iter_number >= n_iters) then sampling_alist else sampling_alist @ (iter (iter_number + 1)) in
   Map.Poly.of_alist_multi (iter 0)
;;

let sorted_var_and_weight_list = List.sort var_and_weight_list ~compare:(fun (x1, x2) (y1, y2)-> Float.compare x1 y1);;
let rec cumsum ~init:accum weights = match weights with 
    [] -> []
    | x::[] -> [accum]
    | x::xs -> accum::(cumsum ~init:(accum +. x) xs)

let summary_statistics_for_var result_map v = 
    let var_and_weight_list = Map.Poly.find_exn result_map v in
    let unnormalized_prob_sum = List.map ~f:(fun x-> Float.exp (snd x)) var_and_weight_list |> List.fold ~init:0. ~f:Float.add in
    let weighted_average = (List.fold ~init:0. ~f:(fun acc (sampled_x, log_w) -> acc +. (sampled_x *. Float.exp(log_w))) var_and_weight_list ) /. unnormalized_prob_sum in
    let weighted_variance = (List.fold ~init:0. var_and_weight_list ~f:(fun acc (sampled_x, log_w) -> acc +. (Float.square (sampled_x -. weighted_average)))) /. unnormalized_prob_sum in
    [("mean", weighted_average); ("variance", weighted_variance)]
    
    (* TODO:
    let log_probs = List.map ~f:(fun x-> -1. *. (snd x)) var_and_weight_list;;
    let unnormalized_prob_logsum = unnormalized_prob_sum |> Float.log;;
    let normalized_probs = List.map ~f:(fun x -> (x -. unnormalized_prob_logsum) }> Float.exp) log_probs;;
    *)

let get_all_summary_statistics result_map = 
    let keys = Map.Poly.keys result_map in
    List.map keys ~f:(fun v -> (v, (summary_statistics_for_var result_map v)))
;;

(* TODO: Wrong, until matplotlib dist can plot with weights... *)
let variable_hist result_map x = 
  Pyplot.hist ((List.map ~f:fst (Map.Poly.find_exn result_map (Variable x))) |> Array.of_list);
  Mpl.show()
;;
