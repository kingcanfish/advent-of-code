mod part1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let (left, right) = part1::read_input("inputs/day1_part1.txt").unwrap();
        let result = part1::resolve(left, right);
        println!("{}", result);
    }
}
