use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

#[allow(unused)]
pub(crate) fn read_input() -> Result<Vec<String>, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("inputs")
        .join("day11.txt");
    let lines: Vec<String> = std::fs::read_to_string(path)?
        .lines()
        .map(|s| s.to_string())
        .collect();
    Ok(lines)
}

/// 解析设备连接关系，构建图结构
fn parse_connections(lines: &[String]) -> HashMap<String, Vec<String>> {
    let mut graph = HashMap::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() != 2 {
            continue;
        }

        let device = parts[0].trim().to_string();
        let outputs: Vec<String> = parts[1]
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        graph.insert(device, outputs);
    }

    graph
}

/// 使用深度优先搜索计算从起点到终点的所有路径数量
fn count_paths(
    graph: &HashMap<String, Vec<String>>,
    current: &str,
    target: &str,
    visited: &mut std::collections::HashSet<String>,
) -> usize {
    // 如果到达目标节点，计数一条路径
    if current == target {
        return 1;
    }

    // 避免循环访问
    if visited.contains(current) {
        return 0;
    }

    // 标记当前节点为已访问
    visited.insert(current.to_string());

    let mut total_paths = 0;

    // 遍历所有出边
    if let Some(outputs) = graph.get(current) {
        for next_device in outputs {
            total_paths += count_paths(graph, next_device, target, visited);
        }
    }

    // 回溯：移除当前节点的访问标记
    visited.remove(current);

    total_paths
}

/// 使用记忆化搜索和状态压缩计算路径数量
fn count_paths_with_devices(
    graph: &HashMap<String, Vec<String>>,
    current: &str,
    target: &str,
    required_devices: &[&str],
) -> usize {
    use std::collections::HashMap;

    // 为必需设备分配位
    let device_to_bit: HashMap<&str, usize> = required_devices
        .iter()
        .enumerate()
        .map(|(i, &device)| (device, i))
        .collect();

    let num_states = 1 << required_devices.len();
    let target_state = num_states - 1; // 所有必需设备都被访问的状态

    // memo[(node, current_state)] = 从此状态到目标的有效路径数量
    let mut memo: HashMap<(String, usize), usize> = HashMap::new();

    fn dfs(
        graph: &HashMap<String, Vec<String>>,
        current: &str,
        target: &str,
        current_state: usize,
        device_to_bit: &HashMap<&str, usize>,
        target_state: usize,
        memo: &mut HashMap<(String, usize), usize>,
    ) -> usize {
        // 如果已经计算过，直接返回
        let key = (current.to_string(), current_state);
        if let Some(&cached) = memo.get(&key) {
            return cached;
        }

        // 如果到达目标节点
        if current == target {
            let result = if current_state == target_state { 1 } else { 0 };
            memo.insert(key, result);
            return result;
        }

        let mut total_paths = 0;

        // 遍历所有出边
        if let Some(outputs) = graph.get(current) {
            for next_device in outputs {
                // 计算访问next_device后的新状态
                let mut next_state = current_state;
                if let Some(&bit) = device_to_bit.get(&next_device.as_str()) {
                    next_state |= 1 << bit;
                }

                total_paths += dfs(
                    graph,
                    next_device,
                    target,
                    next_state,
                    device_to_bit,
                    target_state,
                    memo,
                );
            }
        }

        memo.insert(key, total_paths);
        total_paths
    }

    // 初始化起始状态
    let start_bit = if let Some(&bit) = device_to_bit.get(&current) {
        1 << bit
    } else {
        0
    };

    dfs(
        graph,
        current,
        target,
        start_bit,
        &device_to_bit,
        target_state,
        &mut memo,
    )
}

/// 解决第一部分：从"you"到"out"的所有路径数量
#[allow(unused)]
pub(crate) fn resolve_part1(lines: &[String]) -> Result<usize, Box<dyn Error>> {
    let graph = parse_connections(lines);
    let mut visited = std::collections::HashSet::new();

    let result = count_paths(&graph, "you", "out", &mut visited);
    Ok(result)
}

/// 解决第二部分：找到从"svr"到"out"的所有路径，并筛选出同时经过"dac"和"fft"的路径数量
#[allow(unused)]
pub(crate) fn resolve_part2(lines: &[String]) -> Result<usize, Box<dyn Error>> {
    let graph = parse_connections(lines);

    // 使用记忆化搜索和状态压缩算法
    let required_devices = ["dac", "fft"];
    let result = count_paths_with_devices(&graph, "svr", "out", &required_devices);

    Ok(result)
}

pub(crate) fn run() {
    match read_input() {
        Ok(inputs) => {
            match resolve_part1(&inputs) {
                Ok(result) => print!("Day11 part1: {}, ", result),
                Err(e) => println!("day11 part1 error: {}", e),
            }
            match resolve_part2(&inputs) {
                Ok(result) => println!("Day11 part2: {}", result),
                Err(e) => println!("day11 part2 error: {}", e),
            }
        }
        Err(e) => println!("day11 read input error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_connections() {
        let input = vec![
            "aaa: bbb ccc".to_string(),
            "bbb: ddd eee".to_string(),
            "ccc: ddd eee fff".to_string(),
            "ddd: ggg".to_string(),
            "eee: out".to_string(),
            "fff: out".to_string(),
            "ggg: out".to_string(),
        ];

        let graph = parse_connections(&input);

        assert_eq!(
            graph.get("aaa").unwrap(),
            &vec!["bbb".to_string(), "ccc".to_string()]
        );
        assert_eq!(
            graph.get("bbb").unwrap(),
            &vec!["ddd".to_string(), "eee".to_string()]
        );
        assert_eq!(
            graph.get("ccc").unwrap(),
            &vec!["ddd".to_string(), "eee".to_string(), "fff".to_string()]
        );
    }

    #[test]
    fn test_example_paths() {
        let input = vec![
            "you: bbb ccc".to_string(),
            "bbb: ddd eee".to_string(),
            "ccc: ddd eee fff".to_string(),
            "ddd: ggg".to_string(),
            "eee: out".to_string(),
            "fff: out".to_string(),
            "ggg: out".to_string(),
        ];

        let result = resolve_part1(&input).unwrap();
        // 根据题目描述，应该有5条路径
        assert_eq!(result, 5);
    }

    #[test]
    fn test_no_path() {
        let input = vec![
            "you: aaa".to_string(),
            "aaa: bbb".to_string(),
            "bbb: ccc".to_string(),
            // 没有到"out"的路径
        ];

        let result = resolve_part1(&input).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_single_path() {
        let input = vec!["you: out".to_string()];

        let result = resolve_part1(&input).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_cyclic_graph() {
        let input = vec![
            "you: aaa".to_string(),
            "aaa: bbb".to_string(),
            "bbb: aaa ccc".to_string(),
            "ccc: out".to_string(),
        ];

        let result = resolve_part1(&input).unwrap();
        // 应该只有一条路径，避免循环
        assert_eq!(result, 1);
    }

    #[test]
    fn test_part1_actual() {
        let inputs = read_input().unwrap();
        let result = resolve_part1(&inputs).unwrap();
        println!("Part 1: {}", result);
    }

    #[test]
    fn test_part2_actual() {
        let inputs = read_input().unwrap();
        let result = resolve_part2(&inputs).unwrap();
        println!("Part 2: {}", result);
    }

    #[test]
    fn test_part2_example() {
        // 使用题目中给出的例子
        let input = vec![
            "svr: aaa bbb".to_string(),
            "aaa: fft".to_string(),
            "fft: ccc".to_string(),
            "bbb: tty".to_string(),
            "tty: ccc".to_string(),
            "ccc: ddd eee".to_string(),
            "ddd: hub".to_string(),
            "hub: fff".to_string(),
            "eee: dac".to_string(),
            "dac: fff".to_string(),
            "fff: ggg hhh".to_string(),
            "ggg: out".to_string(),
            "hhh: out".to_string(),
        ];

        let result = resolve_part2(&input).unwrap();
        // 根据题目描述，应该只有2条路径同时经过dac和fft
        assert_eq!(result, 2);
    }
}
