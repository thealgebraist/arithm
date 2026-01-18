(* Proof of 8 scenarios *)

From Coq Require Import List.
From Coq Require Import Arith.
From Coq Require Import BinNat.
From Coq Require Import Bool.

Load "arithmetic.v".

Section Scenarios.

  Open Scope N_scope.

  (* Helper to compare lists of bools *)
  Fixpoint beq_list_bool (l1 l2 : list bool) : bool :=
    match l1, l2 with
    | nil, nil => true
    | b1 :: t1, b2 :: t2 => (eqb b1 b2) && (beq_list_bool t1 t2)
    | _, _ => false
    end.

  Definition bit_to_range (b : bool) : symbol_range :=
    if b then {| cum_low := 50; cum_high := 100; total := 100 |}
         else {| cum_low := 0; cum_high := 50; total := 100 |}.

  Definition range_to_bit (v : N) : bool :=
    if N.ltb v 50 then false else true.

  Definition sym_to_r (b : bool) := bit_to_range b.
  Definition r_to_sym (v : N) := range_to_bit v.
  Definition total_f := 100.

  Fixpoint encode_msg (l : list bool) (s : encoder_state) : encoder_state :=
    match l with
    | nil => s
    | h :: t => encode_msg t (encode_step s (sym_to_r h))
    end.

  Fixpoint decode_msg (len : nat) (s : decoder_state) : list bool :=
    match len with
    | O => nil
    | S n =>
        let r := sym_to_r (r_to_sym (decode_get_scaled_value s total_f)) in
        let s_next := renormalize_decoder {|
          d_low := (d_low s) + (((d_high s) - (d_low s) + 1) * (cum_low r)) / total_f;
          d_high := (d_low s) + (((d_high s) - (d_low s) + 1) * (cum_high r)) / total_f - 1;
          d_value := d_value s;
          d_input := d_input s
        |} 64%nat in
        (r_to_sym (decode_get_scaled_value s total_f)) :: decode_msg n s_next
    end.

  Definition run_test (msg : list bool) : bool :=
    let enc := encode_msg msg init_encoder in
    let bits := finalize_encoder enc in
    let dec := decode_msg (length msg) (init_decoder bits) in
    beq_list_bool msg dec.

  Theorem test1 : run_test (false :: nil) = true. Proof. reflexivity. Qed.
  Theorem test2 : run_test (true :: nil) = true. Proof. reflexivity. Qed.
  Theorem test3 : run_test (false :: true :: nil) = true. Proof. reflexivity. Qed.
  Theorem test4 : run_test (true :: false :: nil) = true. Proof. reflexivity. Qed.
  Theorem test5 : run_test (false :: false :: false :: nil) = true. Proof. reflexivity. Qed.
  Theorem test6 : run_test (true :: true :: true :: nil) = true. Proof. reflexivity. Qed.
  Theorem test7 : run_test (true :: false :: true :: false :: nil) = true. Proof. reflexivity. Qed.
  Theorem test8 : run_test (nil) = true. Proof. reflexivity. Qed.

End Scenarios.
