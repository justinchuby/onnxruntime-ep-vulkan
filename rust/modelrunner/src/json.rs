//! A small JSON reader/writer, in-tree for the same reason as `sha256.rs`: no package index.
//!
//! It has to do four jobs and no more:
//!
//!   * read `bench/results/model_provenance.json` (the pinned-provenance contract);
//!   * read the EP's counters dump (`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE`);
//!   * read ORT's own profiling trace, which is a Chrome Trace array of ~10^4 objects, so the
//!     parser is iterative where it can be and never recurses per array *element*;
//!   * write this runner's evidence artifact, stably ordered so a re-run producing the same
//!     facts produces the same bytes (an artifact frame hashes the file).
//!
//! Object member order is *preserved*, not sorted: the evidence artifact is written in a reading
//! order chosen for humans, and a reader diffing two runs should see field moves only when the
//! writer changed.
//!
//! Numbers are `f64` on the way in, as JSON says. The writer keeps integers integral -- a counter
//! that reads `dispatches_executed: 2.0` in one artifact and `2` in another is a diff nobody
//! should have to explain.

use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub message: String,
    pub offset: usize,
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at byte {})", self.message, self.offset)
    }
}

impl std::error::Error for JsonError {}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// `None` for a non-number *and* for a number that is not integral. A counter that arrived as
    /// `2.5` is not "2": it is a malformed counter, and the caller must be able to tell.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Num(n) if n.fract() == 0.0 && n.is_finite() => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub fn str_of(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Json::as_str)
    }

    pub fn i64_of(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(Json::as_i64)
    }

    pub fn obj(members: Vec<(&str, Json)>) -> Json {
        Json::Obj(
            members
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    pub fn s(text: impl Into<String>) -> Json {
        Json::Str(text.into())
    }

    pub fn n(value: impl Into<f64>) -> Json {
        Json::Num(value.into())
    }

    pub fn int(value: i64) -> Json {
        Json::Num(value as f64)
    }
}

pub fn parse(text: &str) -> Result<Json, JsonError> {
    let bytes = text.as_bytes();
    let mut p = Parser { bytes, pos: 0 };
    p.skip_ws();
    let value = p.value(0)?;
    p.skip_ws();
    if p.pos != bytes.len() {
        return Err(p.err("trailing content after the top-level JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

/// ORT's Chrome Trace is flat, and `model_provenance.json` is two levels deep. Anything deeper
/// than this is either a different file than we think or an attempt to blow the stack.
const MAX_DEPTH: usize = 64;

impl<'a> Parser<'a> {
    fn err(&self, message: &str) -> JsonError {
        JsonError {
            message: message.to_string(),
            offset: self.pos,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, expected: u8) -> Result<(), JsonError> {
        if self.bytes.get(self.pos) == Some(&expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", expected as char)))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth > MAX_DEPTH {
            return Err(self.err("JSON nested deeper than this reader accepts"));
        }
        self.skip_ws();
        match self.bytes.get(self.pos) {
            None => Err(self.err("unexpected end of input")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(_) => self.number(),
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.err(&format!("expected `{word}`")))
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.eat(b'{')?;
        let mut members = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Obj(members));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            let value = self.value(depth + 1)?;
            members.push((key, value));
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(members));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let b = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| self.err("unterminated string"))?;
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| self.err("unterminated escape"))?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            // Surrogate pairs: ORT writes model and node names verbatim, and a
                            // model with an emoji in a node name is legal ONNX. Dropping the low
                            // half would silently corrupt a name we later match on.
                            let ch = if (0xD800..0xDC00).contains(&hi) {
                                if self.bytes.get(self.pos) == Some(&b'\\')
                                    && self.bytes.get(self.pos + 1) == Some(&b'u')
                                {
                                    self.pos += 2;
                                    let lo = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        let cp = 0x10000
                                            + (((hi as u32) - 0xD800) << 10)
                                            + ((lo as u32) - 0xDC00);
                                        char::from_u32(cp)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                char::from_u32(hi as u32)
                            };
                            out.push(ch.ok_or_else(|| self.err("invalid \\u escape"))?);
                        }
                        other => {
                            return Err(self.err(&format!("unknown escape \\{}", other as char)));
                        }
                    }
                }
                _ => {
                    // Copy the UTF-8 sequence through unchanged.
                    let start = self.pos - 1;
                    let len = utf8_len(b);
                    if start + len > self.bytes.len() {
                        return Err(self.err("truncated UTF-8 sequence in string"));
                    }
                    let s = std::str::from_utf8(&self.bytes[start..start + len])
                        .map_err(|_| self.err("invalid UTF-8 in string"))?;
                    out.push_str(s);
                    self.pos = start + len;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u16, JsonError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| self.err("invalid \\u escape"))?;
        let v = u16::from_str_radix(s, 16).map_err(|_| self.err("invalid \\u escape"))?;
        self.pos += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_digit() || b == b'.' || b == b'e' || b == b'E' || b == b'+' || b == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(self.err("expected a JSON value"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("invalid number"))?;
        text.parse::<f64>().map(Json::Num).map_err(|_| JsonError {
            message: format!("`{text}` is not a JSON number"),
            offset: start,
        })
    }
}

fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// Pretty-print with two-space indent and a trailing newline: the shape every other committed
/// artifact in `bench/results/` already has, so a diff against one is readable.
pub fn to_string_pretty(value: &Json) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out.push('\n');
    out
}

fn write_value(out: &mut String, value: &Json, indent: usize) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => out.push_str(&format_number(*n)),
        Json::Str(s) => write_string(out, s),
        Json::Arr(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                pad(out, indent + 1);
                write_value(out, item, indent + 1);
                if i + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            pad(out, indent);
            out.push(']');
        }
        Json::Obj(members) => {
            if members.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (i, (key, item)) in members.iter().enumerate() {
                pad(out, indent + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, indent + 1);
                if i + 1 < members.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            pad(out, indent);
            out.push('}');
        }
    }
}

fn pad(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

/// Integral values print without a fraction; non-finite values print as `null`, because JSON has
/// no NaN and a runner that emitted a bare `NaN` token would write a file no reader can parse.
/// A NaN that matters is reported in its own `*_is_nan` boolean beside the number, never smuggled
/// into it.
fn format_number(n: f64) -> String {
    if !n.is_finite() {
        return "null".to_string();
    }
    if n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 {
        format!("{}", n as i64)
    } else {
        let mut s = format!("{n:?}");
        if s.ends_with(".0") {
            s.truncate(s.len() - 2);
        }
        s
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Count `cat == "Node"` events per `args.provider` in an ORT profiling trace.
///
/// This is a port of `tests/ops/_verdict.py::tally_providers`, deliberately including its
/// robustness rule: a malformed event is skipped rather than fatal. A trace with one bad event is
/// still a trace, and an instrument that dies on unfamiliar input reports an outage that reads
/// like a finding.
pub fn tally_providers(trace: &Json) -> BTreeMap<String, u64> {
    let mut tally = BTreeMap::new();
    let events: &[Json] = match trace {
        Json::Arr(items) => items.as_slice(),
        // Some ORT builds wrap the array; accept `{"traceEvents": [...]}` too.
        Json::Obj(_) => match trace.get("traceEvents").and_then(Json::as_array) {
            Some(items) => items,
            None => return tally,
        },
        _ => return tally,
    };
    for event in events {
        if event.str_of("cat") != Some("Node") {
            continue;
        }
        let Some(args) = event.get("args") else {
            continue;
        };
        let Some(provider) = args.str_of("provider") else {
            continue;
        };
        if provider.is_empty() {
            continue;
        }
        *tally.entry(provider.to_string()).or_insert(0) += 1;
    }
    tally
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_nested_document() {
        let text = r#"{"models":[{"name":"mnist-12","bytes":26143,"sha256":"abc"}],"ok":true}"#;
        let value = parse(text).unwrap();
        assert_eq!(
            value
                .get("models")
                .unwrap()
                .as_array()
                .unwrap()
                .first()
                .unwrap()
                .str_of("name"),
            Some("mnist-12")
        );
        assert_eq!(
            value.get("models").unwrap().as_array().unwrap()[0].i64_of("bytes"),
            Some(26143)
        );
        let printed = to_string_pretty(&value);
        assert_eq!(parse(&printed).unwrap(), value);
    }

    #[test]
    fn integers_stay_integral_through_a_write() {
        let doc = Json::obj(vec![("dispatches_executed", Json::int(2))]);
        assert!(to_string_pretty(&doc).contains("\"dispatches_executed\": 2"));
        assert!(!to_string_pretty(&doc).contains("2.0"));
    }

    #[test]
    fn non_finite_numbers_do_not_produce_unparseable_output() {
        let doc = Json::obj(vec![("max_rel", Json::Num(f64::NAN))]);
        let text = to_string_pretty(&doc);
        assert!(text.contains("null"), "{text}");
        parse(&text).expect("a NaN must not make the artifact unreadable");
    }

    #[test]
    fn malformed_input_is_an_error_with_an_offset() {
        let err = parse("{\"a\": }").unwrap_err();
        assert!(err.offset > 0, "{err}");
        assert!(parse("").is_err());
        assert!(parse("{}{}").is_err());
        assert!(parse("[1, 2").is_err());
    }

    #[test]
    fn escapes_and_unicode_survive() {
        let value = parse(r#"{"k":"a\"b\\c\nd\u00e9\ud83d\ude00"}"#).unwrap();
        assert_eq!(value.str_of("k"), Some("a\"b\\c\ndé😀"));
        let printed = to_string_pretty(&value);
        assert_eq!(parse(&printed).unwrap(), value);
    }

    #[test]
    fn fractional_counter_is_not_silently_truncated() {
        let value = parse(r#"{"n": 2.5}"#).unwrap();
        assert_eq!(value.i64_of("n"), None);
        assert_eq!(value.get("n").unwrap().as_f64(), Some(2.5));
    }

    #[test]
    fn provider_tally_counts_only_node_events() {
        let trace = parse(
            r#"[
              {"cat":"Node","name":"a","args":{"provider":"VulkanExecutionProvider"}},
              {"cat":"Node","name":"b","args":{"provider":"CPUExecutionProvider"}},
              {"cat":"Node","name":"c","args":{"provider":"VulkanExecutionProvider"}},
              {"cat":"Session","name":"model_run","args":{"provider":"VulkanExecutionProvider"}},
              {"cat":"Node","name":"d"},
              {"cat":"Node","name":"e","args":{"provider":""}},
              "not an object"
            ]"#,
        )
        .unwrap();
        let tally = tally_providers(&trace);
        assert_eq!(tally.get("VulkanExecutionProvider"), Some(&2));
        assert_eq!(tally.get("CPUExecutionProvider"), Some(&1));
        assert_eq!(tally.len(), 2);
    }

    #[test]
    fn provider_tally_accepts_the_wrapped_trace_shape() {
        let trace = parse(
            r#"{"traceEvents":[{"cat":"Node","name":"a","args":{"provider":"CPUExecutionProvider"}}]}"#,
        )
        .unwrap();
        assert_eq!(
            tally_providers(&trace).get("CPUExecutionProvider"),
            Some(&1)
        );
    }
}
