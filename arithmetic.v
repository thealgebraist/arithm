(* Arithmetic Coding in Coq *)
(* Constraints: No rewrite, admit, lra, unfold, replace. Full derivations. *)

From Stdlib Require Import List.
From Stdlib Require Import Arith.
From Stdlib Require Import BinNat.
From Stdlib Require Import Bool.

Section ArithmeticCoding.

  Open Scope N_scope.

  Definition precision : N := 32.
  Definition max_code : N := (2 ^ precision) - 1.
  Definition half_code : N := 2 ^ (precision - 1).
  Definition quarter_code : N := 2 ^ (precision - 2).
  Definition three_quarter_code : N := 3 * quarter_code.

  Record symbol_range := {
    cum_low : N;
    cum_high : N;
    total : N
  }.

  Record encoder_state := {
    e_low : N;
    e_high : N;
    e_pending : N;
    e_output : list bool
  }.

  Definition init_encoder : encoder_state := {|
    e_low := 0;
    e_high := max_code;
    e_pending := 0;
    e_output := nil
  |}.

  Fixpoint append_bits (out : list bool) (b : bool) (p : nat) : list bool :=
    match p with
    | O => out ++ (b :: nil)
    | S n => out ++ (b :: repeat (negb b) (S n))
    end.

  Fixpoint renormalize_encoder (s : encoder_state) (count : nat) : encoder_state :=
    match count with
    | O => s
    | S n =>
        let l := e_low s in
        let h := e_high s in
        let p := N.to_nat (e_pending s) in
        let out := e_output s in
        if N.ltb h half_code then
          renormalize_encoder {|
            e_low := (l * 2) mod (max_code + 1);
            e_high := (h * 2 + 1) mod (max_code + 1);
            e_pending := 0;
            e_output := append_bits out false p
          |} n
        else if N.leb half_code l then
          renormalize_encoder {|
            e_low := ((l - half_code) * 2) mod (max_code + 1);
            e_high := (((h - half_code) * 2) + 1) mod (max_code + 1);
            e_pending := 0;
            e_output := append_bits out true p
          |} n
        else if (N.leb quarter_code l) && (N.ltb h three_quarter_code) then
          renormalize_encoder {|
            e_low := ((l - quarter_code) * 2) mod (max_code + 1);
            e_high := (((h - quarter_code) * 2) + 1) mod (max_code + 1);
            e_pending := e_pending s + 1;
            e_output := out
          |} n
        else s
    end.

  Definition encode_step (s : encoder_state) (r : symbol_range) : encoder_state :=
    let range := (e_high s) - (e_low s) + 1 in
    let new_high := (e_low s) + (range * (cum_high r)) / (total r) - 1 in
    let new_low  := (e_low s) + (range * (cum_low r)) / (total r) in
    renormalize_encoder {|
      e_low := new_low;
      e_high := new_high;
      e_pending := e_pending s;
      e_output := e_output s
    |} 64%nat.

  Definition finalize_encoder (s : encoder_state) : list bool :=
    e_output s ++ (true :: repeat false (N.to_nat (e_pending s + 1))).

  Record decoder_state := {
    d_low : N;
    d_high : N;
    d_value : N;
    d_input : list bool
  }.

  Definition pull_bit (input : list bool) : bool * list bool :=
    match input with
    | nil => (false, nil)
    | b :: t => (b, t)
    end.

  Fixpoint renormalize_decoder (s : decoder_state) (count : nat) : decoder_state :=
    match count with
    | O => s
    | S n =>
        let l := d_low s in
        let h := d_high s in
        let v := d_value s in
        let inp := d_input s in
        if N.ltb h half_code then
          let (b, next_inp) := pull_bit inp in
          renormalize_decoder {|
            d_low := (l * 2) mod (max_code + 1);
            d_high := (h * 2 + 1) mod (max_code + 1);
            d_value := (v * 2 + (if b then 1 else 0)) mod (max_code + 1);
            d_input := next_inp
          |} n
        else if N.leb half_code l then
          let b_inp := pull_bit inp in
          let b := fst b_inp in
          let next_inp := snd b_inp in
          renormalize_decoder {|
            d_low := ((l - half_code) * 2) mod (max_code + 1);
            d_high := (((h - half_code) * 2) + 1) mod (max_code + 1);
            d_value := (((v - half_code) * 2) + (if b then 1 else 0)) mod (max_code + 1);
            d_input := next_inp
          |} n
        else if (N.leb quarter_code l) && (N.ltb h three_quarter_code) then
          let b_inp := pull_bit inp in
          let b := fst b_inp in
          let next_inp := snd b_inp in
          renormalize_decoder {|
            d_low := ((l - quarter_code) * 2) mod (max_code + 1);
            d_high := (((h - quarter_code) * 2) + 1) mod (max_code + 1);
            d_value := (((v - quarter_code) * 2) + (if b then 1 else 0)) mod (max_code + 1);
            d_input := next_inp
          |} n
        else s
    end.

  Fixpoint fill_value_nat (input : list bool) (bits : nat) : N * list bool :=
    match bits with
    | O => (0, input)
    | S n =>
        let (b, rest) := pull_bit input in
        let res := fill_value_nat rest n in
        let v := fst res in
        let final_rest := snd res in
        ((if b then 2 ^ (N.of_nat n) else 0) + v, final_rest)
    end.

  Definition init_decoder (input : list bool) : decoder_state :=
    let res := fill_value_nat input (N.to_nat precision) in
    {| d_low := 0; d_high := max_code; d_value := fst res; d_input := snd res |}.

  Definition decode_get_scaled_value (s : decoder_state) (total_f : N) : N :=
    let l := d_low s in
    let h := d_high s in
    let v := d_value s in
    let r := h - l + 1 in
    ((v - l + 1) * total_f - 1) / r.

End ArithmeticCoding.
