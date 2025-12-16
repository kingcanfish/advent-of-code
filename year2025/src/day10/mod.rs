use crate::file::read_file;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Add;
use z3::Optimize;
use z3::ast::Int;

#[derive(Debug, Clone)]
struct Machine {
    target_pattern: Vec<bool>,
    buttons: Vec<Vec<usize>>,
    target_values: Vec<i64>,
}

fn parse_input(content: &str) -> Vec<Machine> {
    content.lines().map(parse_machine).collect()
}

fn parse_machine(line: &str) -> Machine {
    // Extract target pattern between [ and ]
    let pattern_start = line.find('[').unwrap() + 1;
    let pattern_end = line.find(']').unwrap();
    let pattern_str = &line[pattern_start..pattern_end];
    let target_pattern: Vec<bool> = pattern_str.chars().map(|c| c == '#').collect();

    // Extract buttons between ( and )
    let mut buttons = Vec::new();
    let mut pos = pattern_end;
    while let Some(start) = line[pos..].find('(') {
        let actual_start = pos + start + 1;
        let end = line[actual_start..].find(')').unwrap() + actual_start;
        let button_str = &line[actual_start..end];

        let button: Vec<usize> = button_str
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        buttons.push(button);
        pos = end + 1;

        // Stop if we hit the { which marks the start of target values
        if line[pos..].contains('{')
            && !line[pos..line[pos..].find('{').unwrap() + pos].contains('(')
        {
            break;
        }
    }

    // Extract target values between { and }
    let values_start = line.find('{').unwrap() + 1;
    let values_end = line.find('}').unwrap();
    let values_str = &line[values_start..values_end];
    let target_values: Vec<i64> = values_str
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    Machine {
        target_pattern,
        buttons,
        target_values,
    }
}

// Part 1: Bit Operations for Gaussian Elimination over GF(2)
fn solve_part1_machine(machine: &Machine) -> Option<usize> {
    let _n_lights = machine.target_pattern.len();
    let n_buttons = machine.buttons.len();

    // Use bit representation for efficiency
    // Each row is represented as (coefficients | target) in a 64-bit integer
    let mut rows = Vec::<u64>::new();

    // Build matrix with bit representation
    for button_idx in 0..n_buttons {
        for &light_idx in &machine.buttons[button_idx] {
            if rows.len() <= light_idx {
                rows.resize(light_idx + 1, 0);
            }
            rows[light_idx] |= 1u64 << button_idx;
        }
    }

    // Add target column as the highest bit
    for (light_idx, &target) in machine.target_pattern.iter().enumerate() {
        if rows.len() <= light_idx {
            rows.resize(light_idx + 1, 0);
        }
        if target {
            rows[light_idx] |= 1u64 << n_buttons; // Set the target bit
        }
    }

    gaussian_elimination_bitwise(&mut rows, n_buttons)
}

fn gaussian_elimination_bitwise(rows: &mut [u64], n_vars: usize) -> Option<usize> {
    let n_rows = rows.len();
    let target_bit = 1u64 << n_vars;
    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();

    // Forward elimination using bit operations
    for col in 0..n_vars {
        let col_bit = 1u64 << col;

        // Find pivot
        let mut pivot = None;
        for (row, &row_val) in rows.iter().enumerate().skip(pivot_row) {
            if row_val & col_bit != 0 {
                pivot = Some(row);
                break;
            }
        }

        if let Some(pivot_idx) = pivot {
            // Swap rows
            if pivot_idx != pivot_row {
                rows.swap(pivot_idx, pivot_row);
            }

            pivot_cols.push(col);

            // Eliminate all other rows using XOR
            for row in 0..n_rows {
                if row != pivot_row && (rows[row] & col_bit) != 0 {
                    rows[row] ^= rows[pivot_row];
                }
            }

            pivot_row += 1;
        }
    }

    // Check for inconsistency
    for &row in rows.iter() {
        let coeff_part = row & ((1u64 << n_vars) - 1);
        if coeff_part == 0 && (row & target_bit) != 0 {
            return None; // No solution
        }
    }

    // Find free variables
    let mut pivot_vars = vec![false; n_vars];
    for &col in &pivot_cols {
        pivot_vars[col] = true;
    }
    let free_vars: Vec<usize> = (0..n_vars).filter(|&i| !pivot_vars[i]).collect();

    // Try all combinations of free variables to minimize Hamming weight
    let num_free = free_vars.len();
    let mut min_count = usize::MAX;

    for mask in 0..(1u64 << num_free) {
        let mut solution = vec![false; n_vars];

        // Set free variables according to mask
        for (i, &var_idx) in free_vars.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                solution[var_idx] = true;
            }
        }

        // Compute pivot variables using back substitution with bit operations
        for (row_idx, &col) in pivot_cols.iter().enumerate() {
            let row = rows[row_idx];
            let mut val = (row & target_bit) != 0;

            // XOR with contributions from known variables (excluding the pivot variable itself)
            for (var_idx, &var_val) in solution.iter().enumerate() {
                if var_idx != col && var_val {
                    let var_bit = 1u64 << var_idx;
                    if (row & var_bit) != 0 {
                        val ^= true;
                    }
                }
            }
            solution[col] = val;
        }

        // Count ones
        let count = solution.iter().filter(|&&x| x).count();
        min_count = min_count.min(count);
    }

    Some(min_count)
}

// Part 2: Z3 Integer Linear Programming
fn solve_part2_machine(machine: &Machine) -> Option<usize> {
    let n_buttons = machine.buttons.len();

    // Create Z3 context and optimizer (z3 0.19 API)
    let optimizer = Optimize::new();

    // Create button press variables
    let mut button_vars = Vec::new();
    for i in 0..n_buttons {
        let var_name = format!("button_{}", i);
        button_vars.push(Int::new_const(var_name.as_str()));
        // Add non-negative constraint
        optimizer.assert(&button_vars[i].ge(Int::from_i64(0)));
    }

    // Build button index mapping by joltage value
    let mut button_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (btn_idx, button) in machine.buttons.iter().enumerate() {
        for &light_idx in button {
            button_indices.entry(light_idx).or_default().push(btn_idx);
        }
    }

    // Add constraints for each joltage target
    for (&joltage_idx, indices) in &button_indices {
        if joltage_idx < machine.target_values.len() {
            let target_value = machine.target_values[joltage_idx];

            // Create sum of button presses for this joltage
            let mut sum_ast = Int::from_i64(0);
            for &btn_idx in indices {
                sum_ast = sum_ast.add(&button_vars[btn_idx]);
            }

            // Add equality constraint (use eq instead of _eq)
            optimizer.assert(&sum_ast.eq(Int::from_i64(target_value)));
        }
    }

    // Minimize total button presses
    let mut total_presses_ast = Int::from_i64(0);
    for btn_var in &button_vars {
        total_presses_ast = total_presses_ast.add(btn_var);
    }
    optimizer.minimize(&total_presses_ast);

    // Check if satisfiable and get solution
    match optimizer.check(&[]) {
        z3::SatResult::Sat => {
            let model = optimizer.get_model().unwrap();
            let total_presses = model
                .eval(&total_presses_ast, true)
                .unwrap()
                .as_i64()
                .unwrap() as usize;
            Some(total_presses)
        }
        _ => None,
    }
}

fn resolve_part1(input: &str) -> Result<i64, Box<dyn Error>> {
    let machines = parse_input(input);
    let total: usize = machines.iter().filter_map(solve_part1_machine).sum();
    Ok(total as i64)
}

fn resolve_part2(input: &str) -> Result<i64, Box<dyn Error>> {
    let machines = parse_input(input);
    let total: usize = machines.iter().filter_map(solve_part2_machine).sum();
    Ok(total as i64)
}

pub(crate) fn run() {
    let input = read_file("day10.txt").unwrap();
    match resolve_part1(&input) {
        Ok(result) => print!("Day10 part1: {},", result),
        Err(e) => println!("day10 part1 error: {}", e),
    }
    match resolve_part2(&input) {
        Ok(result) => print!(" part2: {}", result),
        Err(e) => println!("day10 part2 error: {}", e),
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let line = "[##] (0) (1) {5,3}";
        let machine = parse_machine(line);
        assert_eq!(machine.target_pattern, vec![true, true]);
        assert_eq!(machine.buttons.len(), 2);
        assert_eq!(machine.target_values, vec![5, 3]);
    }

    #[test]
    fn test_part1_example1() {
        let input = "[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}";
        let machine = parse_machine(input);
        let result = solve_part1_machine(&machine);
        assert_eq!(result, Some(2));
    }

    #[test]
    fn test_part1_example2() {
        let input = "[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}";
        let machine = parse_machine(input);
        let result = solve_part1_machine(&machine);
        assert_eq!(result, Some(3));
    }

    #[test]
    fn test_part1_example3() {
        let input = "[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}";
        let machine = parse_machine(input);
        let result = solve_part1_machine(&machine);
        assert_eq!(result, Some(2));
    }
}
