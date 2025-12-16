use crate::file::read_file;
use std::error::Error;

#[derive(Debug)]
struct Range {
    start: i64,
    end: i64,
}

impl Range {
    fn contains(&self, value: i64) -> bool {
        value >= self.start && value <= self.end
    }
}

#[derive(Debug)]
struct Input {
    ranges: Vec<Range>,
    ingredients: Vec<i64>,
}

fn unmarshal_input(content: &str) -> Result<Input, Box<dyn Error>> {
    let parts: Vec<&str> = content.split("\n\n").collect();

    if parts.len() != 2 {
        return Err("Invalid input format".into());
    }

    // Parse ranges
    let ranges: Vec<Range> = parts[0]
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let nums: Vec<i64> = line
                .split('-')
                .map(|s| s.trim().parse::<i64>().unwrap())
                .collect();
            Range {
                start: nums[0],
                end: nums[1],
            }
        })
        .collect();

    // Parse ingredient IDs
    let ingredients: Vec<i64> = parts[1]
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().parse::<i64>().unwrap())
        .collect();

    Ok(Input {
        ranges,
        ingredients,
    })
}

fn resolve_part1(input: &str) -> i64 {
    let data = unmarshal_input(input).unwrap();

    let mut fresh_count = 0;

    for ingredient_id in &data.ingredients {
        // Check if this ingredient ID falls into any range
        let is_fresh = data
            .ranges
            .iter()
            .any(|range| range.contains(*ingredient_id));
        if is_fresh {
            fresh_count += 1;
        }
    }

    fresh_count
}

fn resolve_part2(input: &str) -> i64 {
    let data = unmarshal_input(input).unwrap();

    // Sort ranges by start position
    let mut ranges = data.ranges;
    ranges.sort_by_key(|r| r.start);

    // Merge overlapping ranges
    let mut merged_ranges = Vec::new();
    for range in ranges {
        if merged_ranges.is_empty() {
            merged_ranges.push(range);
        } else {
            let last = merged_ranges.last_mut().unwrap();
            // Check if current range overlaps or is adjacent to last merged range
            if range.start <= last.end + 1 {
                // Merge by extending the end if needed
                last.end = last.end.max(range.end);
            } else {
                // No overlap, add as new range
                merged_ranges.push(range);
            }
        }
    }

    // Count total IDs in all merged ranges
    merged_ranges.iter().map(|r| r.end - r.start + 1).sum()
}

pub(crate) fn run() {
    let input = read_file("day5.txt").unwrap();
    println!(
        "Day 5 Part 1: {}, Part 2: {}",
        resolve_part1(&input),
        resolve_part2(&input)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let input = r#"3-5
10-14
16-20
12-18

1
5
8
11
17
32"#
        .to_string();
        assert_eq!(resolve_part1(&input), 3);
    }

    #[test]
    fn test_part2() {
        let input = r#"3-5
10-14
16-20
12-18

1
5
8
11
17
32"#
        .to_string();
        assert_eq!(resolve_part2(&input), 14);
    }
}
