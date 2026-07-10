use std::fmt::{self, Display};
use std::sync::Arc;

use crate::coding::{Decode, DecodeError, Encode, EncodeError};

/// An owned version of [`Path`] with a `'static` lifetime.
pub type PathOwned = Path<'static>;

/// A trait for types that can be converted to a `Path`.
///
/// When providing a String/str, any leading/trailing slashes are trimmed and multiple consecutive slashes are collapsed.
/// When already a Path, normalization is skipped and the underlying parts are reused without copying.
pub trait AsPath {
	fn as_path(&self) -> Path<'_>;
}

impl<'a> AsPath for &'a str {
	fn as_path(&self) -> Path<'a> {
		Path::new(self)
	}
}

impl<'a> AsPath for &'a Path<'a> {
	fn as_path(&self) -> Path<'a> {
		// We don't normalize again nor do we copy the bytes.
		self.borrow()
	}
}

impl AsPath for Path<'_> {
	fn as_path(&self) -> Path<'_> {
		self.borrow()
	}
}

impl AsPath for String {
	fn as_path(&self) -> Path<'_> {
		Path::new(self)
	}
}

impl<'a> AsPath for &'a String {
	fn as_path(&self) -> Path<'a> {
		Path::new(self)
	}
}

/// A borrowed, unparsed string or a suffix of a shared list of shared parts.
///
/// The `Parts` variant is what makes owned paths cheap: every part is its own
/// reference-counted string, so paths built by joining a common root (the way a
/// relay scopes each session to a customer prefix) share the root's part
/// allocations instead of duplicating the prefix bytes per path. Cloning bumps
/// one refcount on the list and suffix operations only advance `start`.
#[derive(Clone)]
enum Repr<'a> {
	Str(&'a str),
	Parts { list: Arc<[Arc<str>]>, start: usize },
}

/// A broadcast path: a sequence of non-empty parts, like a filesystem path.
///
/// Provides safe prefix matching operations that respect part boundaries,
/// preventing issues like "foo" matching "foobar".
///
/// Paths are automatically trimmed of leading and trailing slashes on creation,
/// making all slashes implicit at boundaries.
/// All paths are RELATIVE; you cannot join with a leading slash to make an absolute path.
///
/// Owned paths ([`PathOwned`]) store each part as a shared reference-counted string:
/// cloning, [`Path::to_owned`] on an owned path, and suffix operations like
/// [`Path::strip_prefix`] do not copy any bytes, and paths built with [`Path::join`]
/// share the parts of both inputs.
///
/// # Examples
/// ```
/// use moq_net::{Path};
///
/// // Creation automatically trims slashes
/// let path1 = Path::new("/foo/bar/");
/// let path2 = Path::new("foo/bar");
/// assert_eq!(path1, path2);
///
/// // Methods accept both &str and Path
/// let base = Path::new("api/v1");
/// assert!(base.has_prefix("api"));
/// assert!(base.has_prefix(&Path::new("api/v1")));
///
/// let joined = base.join("users");
/// assert_eq!(joined, "api/v1/users");
/// ```
#[derive(Clone)]
pub struct Path<'a>(Repr<'a>);

/// Iterator over the parts of a [`Path`], yielded in order.
pub struct Parts<'p> {
	inner: PartsRepr<'p>,
}

enum PartsRepr<'p> {
	Str(std::str::Split<'p, char>),
	List(std::slice::Iter<'p, Arc<str>>),
}

impl<'p> Iterator for Parts<'p> {
	type Item = &'p str;

	fn next(&mut self) -> Option<&'p str> {
		match &mut self.inner {
			// Normalized strings have no empty parts, except that splitting the
			// empty path yields one empty item; skip it.
			PartsRepr::Str(split) => split.by_ref().find(|part| !part.is_empty()),
			PartsRepr::List(iter) => iter.next().map(|part| part.as_ref()),
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		match &self.inner {
			PartsRepr::Str(_) => (0, None),
			PartsRepr::List(iter) => iter.size_hint(),
		}
	}
}

impl<'a> Path<'a> {
	/// Maximum number of slash-separated parts in a path.
	///
	/// Matches the IETF moq-transport limit of 32 fields in a namespace tuple.
	/// moq-lite enforces the same bound: encoding or decoding a deeper path fails,
	/// and publishing one to an origin is rejected.
	pub const MAX_PARTS: usize = 32;

	/// Create a new Path from a string slice.
	///
	/// Leading and trailing slashes are automatically trimmed.
	/// Multiple consecutive internal slashes are collapsed to single slashes.
	pub fn new(s: &'a str) -> Self {
		let trimmed = s.trim_start_matches('/').trim_end_matches('/');
		Self(Repr::Str(trimmed))
	}

	/// Iterate over the parts of the path. The empty path has no parts.
	///
	/// # Examples
	/// ```
	/// use moq_net::Path;
	///
	/// let path = Path::new("foo/bar/baz");
	/// assert_eq!(path.parts().collect::<Vec<_>>(), ["foo", "bar", "baz"]);
	/// assert_eq!(Path::empty().parts().count(), 0);
	/// ```
	pub fn parts(&self) -> Parts<'_> {
		Parts {
			inner: match &self.0 {
				Repr::Str(s) => PartsRepr::Str(s.split('/')),
				Repr::Parts { list, start } => PartsRepr::List(list[*start..].iter()),
			},
		}
	}

	/// Get the nth part of the path, if any.
	pub fn part(&self, n: usize) -> Option<&str> {
		self.parts().nth(n)
	}

	/// Number of parts in the path.
	pub fn part_count(&self) -> usize {
		match &self.0 {
			Repr::Str(_) => self.parts().count(),
			Repr::Parts { list, start } => list.len() - start,
		}
	}

	/// Check if this path has the given prefix, respecting part boundaries.
	///
	/// Unlike String::starts_with, this ensures that "foo" does not match "foobar".
	/// The prefix must either:
	/// - Be exactly equal to this path
	/// - Be a leading sequence of this path's parts
	/// - Be empty (matches everything)
	///
	/// # Examples
	/// ```
	/// use moq_net::Path;
	///
	/// let path = Path::new("foo/bar");
	/// assert!(path.has_prefix("foo"));
	/// assert!(path.has_prefix(&Path::new("foo")));
	/// assert!(path.has_prefix("foo/"));
	/// assert!(!path.has_prefix("fo"));
	///
	/// let path = Path::new("foobar");
	/// assert!(!path.has_prefix("foo"));
	/// ```
	pub fn has_prefix(&self, prefix: impl AsPath) -> bool {
		let prefix = prefix.as_path();
		let mut parts = self.parts();

		for expected in prefix.parts() {
			if parts.next() != Some(expected) {
				return false;
			}
		}

		true
	}

	// Split a borrowed string into its first part and the rest, both borrowed.
	fn split_first(s: &str) -> Option<(&str, &str)> {
		let dir = s.split('/').find(|part| !part.is_empty())?;
		// `dir` is a subslice of `s`, so pointer arithmetic finds where it ends.
		let offset = dir.as_ptr() as usize - s.as_ptr() as usize + dir.len();
		Some((dir, s[offset..].trim_start_matches('/')))
	}

	/// Strip the given prefix, returning the rest of the path.
	///
	/// Returns None if the prefix doesn't match according to [`Self::has_prefix`] rules.
	pub fn strip_prefix(&'a self, prefix: impl AsPath) -> Option<Path<'a>> {
		let prefix = prefix.as_path();

		match &self.0 {
			Repr::Str(s) => {
				// Walk the prefix parts along the string so the rest stays borrowed.
				let mut rest: &'a str = s;
				for expected in prefix.parts() {
					let (dir, next) = Self::split_first(rest)?;
					if dir != expected {
						return None;
					}
					rest = next;
				}
				Some(Path(Repr::Str(rest)))
			}
			Repr::Parts { list, start } => {
				let mut parts = self.parts();
				let mut count = 0;
				for expected in prefix.parts() {
					if parts.next() != Some(expected) {
						return None;
					}
					count += 1;
				}
				Some(Path(Repr::Parts {
					list: list.clone(),
					start: start + count,
				}))
			}
		}
	}

	/// Strip the first part of the path, if any, and return it with the rest of the path.
	pub fn next_part(&'a self) -> Option<(&'a str, Path<'a>)> {
		match &self.0 {
			Repr::Str(s) => {
				let (dir, rest) = Self::split_first(s)?;
				Some((dir, Path(Repr::Str(rest))))
			}
			Repr::Parts { list, start } => {
				let dir = list.get(*start)?;
				Some((
					dir.as_ref(),
					Path(Repr::Parts {
						list: list.clone(),
						start: start + 1,
					}),
				))
			}
		}
	}

	/// The empty path, a prefix of every path.
	pub fn empty() -> Path<'static> {
		Path(Repr::Str(""))
	}

	/// Whether the path has no parts.
	pub fn is_empty(&self) -> bool {
		self.parts().next().is_none()
	}

	/// The length of the path in bytes when displayed, including separators.
	pub fn len(&self) -> usize {
		let mut len = 0;
		for part in self.parts() {
			len += part.len() + 1;
		}
		len.saturating_sub(1)
	}

	/// Convert to an owned path, parsing borrowed strings into shared parts.
	pub fn to_owned(&self) -> PathOwned {
		match &self.0 {
			Repr::Str(s) if s.is_empty() => Path::empty(),
			Repr::Str(_) => Path(Repr::Parts {
				list: self.parts().map(Arc::from).collect(),
				start: 0,
			}),
			Repr::Parts { list, start } => Path(Repr::Parts {
				list: list.clone(),
				start: *start,
			}),
		}
	}

	/// Convert into an owned path, parsing borrowed strings into shared parts.
	pub fn into_owned(self) -> PathOwned {
		self.to_owned()
	}

	/// A copy of this path bound to `self`'s lifetime, without copying any bytes.
	pub fn borrow(&'a self) -> Path<'a> {
		match &self.0 {
			Repr::Str(s) => Path(Repr::Str(s)),
			Repr::Parts { list, start } => Path(Repr::Parts {
				list: list.clone(),
				start: *start,
			}),
		}
	}

	/// Join this path with another path, sharing the parts of both inputs.
	///
	/// # Examples
	/// ```
	/// use moq_net::Path;
	///
	/// let base = Path::new("foo");
	/// let joined = base.join("bar");
	/// assert_eq!(joined, "foo/bar");
	///
	/// let joined = base.join(&Path::new("bar"));
	/// assert_eq!(joined, "foo/bar");
	/// ```
	pub fn join(&self, other: impl AsPath) -> PathOwned {
		let other = other.as_path();

		if self.is_empty() {
			return other.to_owned();
		}
		if other.is_empty() {
			return self.to_owned();
		}

		let mut list: Vec<Arc<str>> = Vec::with_capacity(self.part_count() + other.part_count());
		for path in [self, &other] {
			match &path.0 {
				// Borrowed parts have no allocation to share; copy each once.
				Repr::Str(_) => list.extend(path.parts().map(Arc::from)),
				// Owned parts are shared with the new path.
				Repr::Parts { list: parts, start } => list.extend(parts[*start..].iter().cloned()),
			}
		}
		Path(Repr::Parts {
			list: list.into(),
			start: 0,
		})
	}
}

// Comparisons, ordering, and hashing are all part-wise so a borrowed and an owned
// path with the same content behave identically (e.g. as map keys). Part-wise and
// joined-string equality agree; ordering is lexicographic over the part sequence.
impl<'b> PartialEq<Path<'b>> for Path<'_> {
	fn eq(&self, other: &Path<'b>) -> bool {
		self.parts().eq(other.parts())
	}
}

impl Eq for Path<'_> {}

impl PartialEq<&str> for Path<'_> {
	fn eq(&self, other: &&str) -> bool {
		*self == Path::new(other)
	}
}

impl PartialEq<str> for Path<'_> {
	fn eq(&self, other: &str) -> bool {
		*self == Path::new(other)
	}
}

impl PartialEq<Path<'_>> for &str {
	fn eq(&self, other: &Path<'_>) -> bool {
		other == self
	}
}

impl PartialOrd for Path<'_> {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for Path<'_> {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.parts().cmp(other.parts())
	}
}

impl std::hash::Hash for Path<'_> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		// str's Hash is self-delimiting, so hashing each part in order is unambiguous.
		for part in self.parts() {
			part.hash(state);
		}
	}
}

impl fmt::Debug for Path<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Path(\"{self}\")")
	}
}

#[cfg(feature = "serde")]
impl serde::Serialize for Path<'_> {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		serializer.collect_str(self)
	}
}

impl<'a> From<&'a str> for Path<'a> {
	fn from(s: &'a str) -> Self {
		Self::new(s)
	}
}

impl<'a> From<&'a String> for Path<'a> {
	fn from(s: &'a String) -> Self {
		Self::new(s)
	}
}

impl Default for Path<'_> {
	fn default() -> Self {
		Path::empty()
	}
}

impl From<String> for Path<'_> {
	fn from(s: String) -> Self {
		Path::new(&s).into_owned()
	}
}

impl Display for Path<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let mut first = true;
		for part in self.parts() {
			if !first {
				write!(f, "/")?;
			}
			first = false;
			write!(f, "{part}")?;
		}
		Ok(())
	}
}

impl<V: Copy> Decode<V> for Path<'_>
where
	String: Decode<V>,
{
	fn decode<R: bytes::Buf>(r: &mut R, version: V) -> Result<Self, DecodeError> {
		let path: Path = String::decode(r, version)?.into();
		if path.part_count() > Path::MAX_PARTS {
			return Err(DecodeError::BoundsExceeded);
		}
		Ok(path)
	}
}

impl<V: Copy> Encode<V> for Path<'_>
where
	for<'a> &'a str: Encode<V>,
{
	fn encode<W: bytes::BufMut>(&self, w: &mut W, version: V) -> Result<(), EncodeError> {
		if self.part_count() > Path::MAX_PARTS {
			return Err(EncodeError::BoundsExceeded);
		}
		self.to_string().as_str().encode(w, version)?;
		Ok(())
	}
}

// A custom deserializer is needed in order to sanitize
#[cfg(feature = "serde")]
impl<'de: 'a, 'a> serde::Deserialize<'de> for Path<'a> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let s = <&'a str as serde::Deserialize<'de>>::deserialize(deserializer)?;
		Ok(Path::new(s))
	}
}

/// A deduplicated list of path prefixes.
///
/// Automatically removes exact duplicates and overlapping prefixes on construction.
/// For example, `["demo", "demo/foo", "anon"]` becomes `["demo", "anon"]` since
/// `"demo"` already covers `"demo/foo"`.
#[derive(Debug, Clone, Default, Eq)]
pub struct PathPrefixes {
	paths: Vec<PathOwned>,
}

impl PathPrefixes {
	/// Create a new PathPrefixes, deduplicating and removing overlapping prefixes.
	///
	/// Shorter prefixes subsume longer ones: `"demo"` covers `"demo/foo"`.
	///
	/// Accepts anything iterable over path-like items:
	/// ```
	/// use moq_net::PathPrefixes;
	///
	/// let list = PathPrefixes::new(["demo", "demo/foo", "anon"]);
	/// assert_eq!(list.len(), 2); // "demo/foo" subsumed by "demo"
	/// ```
	pub fn new(paths: impl IntoIterator<Item = impl AsPath>) -> Self {
		let mut paths: Vec<PathOwned> = paths.into_iter().map(|p| p.as_path().to_owned()).collect();

		if paths.len() <= 1 {
			return Self { paths };
		}

		// Sort by length so shorter (more permissive) prefixes come first.
		// Tie-break lexicographically for canonical ordering.
		paths.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
		paths.dedup();

		let mut result: Vec<PathOwned> = Vec::new();
		'outer: for path in paths {
			for existing in &result {
				if path.has_prefix(existing) {
					continue 'outer;
				}
			}
			result.push(path);
		}

		Self { paths: result }
	}

	pub fn is_empty(&self) -> bool {
		self.paths.is_empty()
	}

	pub fn len(&self) -> usize {
		self.paths.len()
	}

	pub fn iter(&self) -> std::slice::Iter<'_, PathOwned> {
		self.paths.iter()
	}
}

impl std::ops::Deref for PathPrefixes {
	type Target = [PathOwned];

	fn deref(&self) -> &[PathOwned] {
		&self.paths
	}
}

impl FromIterator<PathOwned> for PathPrefixes {
	fn from_iter<I: IntoIterator<Item = PathOwned>>(iter: I) -> Self {
		Self::new(iter)
	}
}

impl From<Vec<PathOwned>> for PathPrefixes {
	fn from(paths: Vec<PathOwned>) -> Self {
		Self::new(paths)
	}
}

impl<'a> PartialEq<Vec<Path<'a>>> for PathPrefixes {
	fn eq(&self, other: &Vec<Path<'a>>) -> bool {
		self.paths == *other
	}
}

impl<'a> PartialEq<PathPrefixes> for Vec<Path<'a>> {
	fn eq(&self, other: &PathPrefixes) -> bool {
		*self == other.paths
	}
}

impl PartialEq for PathPrefixes {
	fn eq(&self, other: &Self) -> bool {
		self.paths == other.paths
	}
}

impl IntoIterator for PathPrefixes {
	type Item = PathOwned;
	type IntoIter = std::vec::IntoIter<PathOwned>;

	fn into_iter(self) -> Self::IntoIter {
		self.paths.into_iter()
	}
}

impl<'a> IntoIterator for &'a PathPrefixes {
	type Item = &'a PathOwned;
	type IntoIter = std::slice::Iter<'a, PathOwned>;

	fn into_iter(self) -> Self::IntoIter {
		self.paths.iter()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_has_prefix() {
		let path = Path::new("foo/bar/baz");

		// Valid prefixes - test with both &str and &Path
		assert!(path.has_prefix(""));
		assert!(path.has_prefix("foo"));
		assert!(path.has_prefix(Path::new("foo")));
		assert!(path.has_prefix("foo/"));
		assert!(path.has_prefix("foo/bar"));
		assert!(path.has_prefix(Path::new("foo/bar/")));
		assert!(path.has_prefix("foo/bar/baz"));

		// Invalid prefixes - should not match partial components
		assert!(!path.has_prefix("f"));
		assert!(!path.has_prefix(Path::new("fo")));
		assert!(!path.has_prefix("foo/b"));
		assert!(!path.has_prefix("foo/ba"));
		assert!(!path.has_prefix(Path::new("foo/bar/ba")));

		// Edge case: "foobar" should not match "foo"
		let path = Path::new("foobar");
		assert!(!path.has_prefix("foo"));
		assert!(path.has_prefix(Path::new("foobar")));
	}

	#[test]
	fn test_strip_prefix() {
		let path = Path::new("foo/bar/baz");

		// Test with both &str and &Path
		assert_eq!(path.strip_prefix("").unwrap(), "foo/bar/baz");
		assert_eq!(path.strip_prefix("foo").unwrap(), "bar/baz");
		assert_eq!(path.strip_prefix(Path::new("foo/")).unwrap(), "bar/baz");
		assert_eq!(path.strip_prefix("foo/bar").unwrap(), "baz");
		assert_eq!(path.strip_prefix(Path::new("foo/bar/")).unwrap(), "baz");
		assert_eq!(path.strip_prefix("foo/bar/baz").unwrap(), "");

		// Should fail for invalid prefixes
		assert!(path.strip_prefix("fo").is_none());
		assert!(path.strip_prefix(Path::new("bar")).is_none());
	}

	#[test]
	fn test_join() {
		// Test with both &str and &Path
		assert_eq!(Path::new("foo").join("bar"), "foo/bar");
		assert_eq!(Path::new("foo/").join(Path::new("bar")), "foo/bar");
		assert_eq!(Path::new("").join("bar"), "bar");
		assert_eq!(Path::new("foo/bar").join(Path::new("baz")), "foo/bar/baz");
	}

	#[test]
	fn test_empty() {
		let empty = Path::new("");
		assert!(empty.is_empty());
		assert_eq!(empty.len(), 0);

		let non_empty = Path::new("foo");
		assert!(!non_empty.is_empty());
		assert_eq!(non_empty.len(), 3);
	}

	#[test]
	fn test_from_conversions() {
		let path1 = Path::from("foo/bar");
		let path2 = Path::from("foo/bar");
		let s = String::from("foo/bar");
		let path3 = Path::from(&s);

		assert_eq!(path1, "foo/bar");
		assert_eq!(path2, "foo/bar");
		assert_eq!(path3, "foo/bar");
	}

	#[test]
	fn test_path_prefix_join() {
		let prefix = Path::new("foo");
		let suffix = Path::new("bar/baz");
		let path = prefix.join(&suffix);
		assert_eq!(path, "foo/bar/baz");

		let prefix = Path::new("foo/");
		let suffix = Path::new("bar/baz");
		let path = prefix.join(&suffix);
		assert_eq!(path, "foo/bar/baz");

		let prefix = Path::new("foo");
		let suffix = Path::new("/bar/baz");
		let path = prefix.join(&suffix);
		assert_eq!(path, "foo/bar/baz");

		let prefix = Path::new("");
		let suffix = Path::new("bar/baz");
		let path = prefix.join(&suffix);
		assert_eq!(path, "bar/baz");
	}

	#[test]
	fn test_path_prefix_conversions() {
		let prefix1 = Path::from("foo/bar");
		let prefix2 = Path::from(String::from("foo/bar"));
		let s = String::from("foo/bar");
		let prefix3 = Path::from(&s);

		assert_eq!(prefix1, "foo/bar");
		assert_eq!(prefix2, "foo/bar");
		assert_eq!(prefix3, "foo/bar");
	}

	#[test]
	fn test_path_suffix_conversions() {
		let suffix1 = Path::from("foo/bar");
		let suffix2 = Path::from(String::from("foo/bar"));
		let s = String::from("foo/bar");
		let suffix3 = Path::from(&s);

		assert_eq!(suffix1, "foo/bar");
		assert_eq!(suffix2, "foo/bar");
		assert_eq!(suffix3, "foo/bar");
	}

	#[test]
	fn test_path_types_basic_operations() {
		let prefix = Path::new("foo/bar");
		assert_eq!(prefix, "foo/bar");
		assert!(!prefix.is_empty());
		assert_eq!(prefix.len(), 7);

		let suffix = Path::new("baz/qux");
		assert_eq!(suffix, "baz/qux");
		assert!(!suffix.is_empty());
		assert_eq!(suffix.len(), 7);

		let empty_prefix = Path::new("");
		assert!(empty_prefix.is_empty());
		assert_eq!(empty_prefix.len(), 0);

		let empty_suffix = Path::new("");
		assert!(empty_suffix.is_empty());
		assert_eq!(empty_suffix.len(), 0);
	}

	#[test]
	fn test_prefix_has_prefix() {
		// Test empty prefix (should match everything)
		let prefix = Path::new("foo/bar");
		assert!(prefix.has_prefix(""));

		// Test exact matches
		let prefix = Path::new("foo/bar");
		assert!(prefix.has_prefix("foo/bar"));

		// Test valid prefixes
		assert!(prefix.has_prefix("foo"));
		assert!(prefix.has_prefix("foo/"));

		// Test invalid prefixes - partial matches should fail
		assert!(!prefix.has_prefix("f"));
		assert!(!prefix.has_prefix("fo"));
		assert!(!prefix.has_prefix("foo/b"));
		assert!(!prefix.has_prefix("foo/ba"));

		// Test edge cases
		let prefix = Path::new("foobar");
		assert!(!prefix.has_prefix("foo"));
		assert!(prefix.has_prefix("foobar"));

		// Test trailing slash handling
		let prefix = Path::new("foo/bar/");
		assert!(prefix.has_prefix("foo"));
		assert!(prefix.has_prefix("foo/"));
		assert!(prefix.has_prefix("foo/bar"));
		assert!(prefix.has_prefix("foo/bar/"));

		// Test single component
		let prefix = Path::new("foo");
		assert!(prefix.has_prefix(""));
		assert!(prefix.has_prefix("foo"));
		assert!(prefix.has_prefix("foo/")); // "foo/" becomes "foo" after trimming
		assert!(!prefix.has_prefix("f"));

		// Test empty prefix
		let prefix = Path::new("");
		assert!(prefix.has_prefix(""));
		assert!(!prefix.has_prefix("foo"));
	}

	#[test]
	fn test_prefix_join() {
		// Basic joining
		let prefix = Path::new("foo");
		let suffix = Path::new("bar");
		assert_eq!(prefix.join(suffix), "foo/bar");

		// Trailing slash on prefix
		let prefix = Path::new("foo/");
		let suffix = Path::new("bar");
		assert_eq!(prefix.join(suffix), "foo/bar");

		// Leading slash on suffix
		let prefix = Path::new("foo");
		let suffix = Path::new("/bar");
		assert_eq!(prefix.join(suffix), "foo/bar");

		// Trailing slash on suffix
		let prefix = Path::new("foo");
		let suffix = Path::new("bar/");
		assert_eq!(prefix.join(suffix), "foo/bar"); // trailing slash is trimmed

		// Both have slashes
		let prefix = Path::new("foo/");
		let suffix = Path::new("/bar");
		assert_eq!(prefix.join(suffix), "foo/bar");

		// Empty suffix
		let prefix = Path::new("foo");
		let suffix = Path::new("");
		assert_eq!(prefix.join(suffix), "foo");

		// Empty prefix
		let prefix = Path::new("");
		let suffix = Path::new("bar");
		assert_eq!(prefix.join(suffix), "bar");

		// Both empty
		let prefix = Path::new("");
		let suffix = Path::new("");
		assert_eq!(prefix.join(suffix), "");

		// Complex paths
		let prefix = Path::new("foo/bar");
		let suffix = Path::new("baz/qux");
		assert_eq!(prefix.join(suffix), "foo/bar/baz/qux");

		// Complex paths with slashes
		let prefix = Path::new("foo/bar/");
		let suffix = Path::new("/baz/qux/");
		assert_eq!(prefix.join(suffix), "foo/bar/baz/qux"); // all slashes are trimmed
	}

	#[test]
	fn test_path_ref() {
		// Test PathRef creation and normalization
		let ref1 = Path::new("/foo/bar/");
		assert_eq!(ref1, "foo/bar");

		let ref2 = Path::from("///foo///");
		assert_eq!(ref2, "foo");

		// Test PathRef normalizes multiple slashes
		let ref3 = Path::new("foo//bar///baz");
		assert_eq!(ref3, "foo/bar/baz");

		// Test conversions
		let path = Path::new("foo/bar");
		let path_ref = path;
		assert_eq!(path_ref, "foo/bar");

		// Test that Path methods work with PathRef
		let path2 = Path::new("foo/bar/baz");
		assert!(path2.has_prefix(&path_ref));
		assert_eq!(path2.strip_prefix(path_ref).unwrap(), "baz");

		// Test empty PathRef
		let empty = Path::new("");
		assert!(empty.is_empty());
		assert_eq!(empty.len(), 0);
	}

	#[test]
	fn test_multiple_consecutive_slashes() {
		let path = Path::new("foo//bar///baz");
		// Multiple consecutive slashes are collapsed to single slashes
		assert_eq!(path, "foo/bar/baz");

		// Test with leading and trailing slashes too
		let path2 = Path::new("//foo//bar///baz//");
		assert_eq!(path2, "foo/bar/baz");

		// Test empty segments are handled correctly
		let path3 = Path::new("foo///bar");
		assert_eq!(path3, "foo/bar");
	}

	#[test]
	fn test_removes_multiple_slashes_comprehensively() {
		// Test various multiple slash scenarios
		assert_eq!(Path::new("foo//bar"), "foo/bar");
		assert_eq!(Path::new("foo///bar"), "foo/bar");
		assert_eq!(Path::new("foo////bar"), "foo/bar");

		// Multiple occurrences of double slashes
		assert_eq!(Path::new("foo//bar//baz"), "foo/bar/baz");
		assert_eq!(Path::new("a//b//c//d"), "a/b/c/d");

		// Mixed slash counts
		assert_eq!(Path::new("foo//bar///baz////qux"), "foo/bar/baz/qux");

		// With leading and trailing slashes
		assert_eq!(Path::new("//foo//bar//"), "foo/bar");
		assert_eq!(Path::new("///foo///bar///"), "foo/bar");

		// Edge case: only slashes
		assert_eq!(Path::new("//"), "");
		assert_eq!(Path::new("////"), "");

		// Test that operations work correctly with normalized paths
		let path_with_slashes = Path::new("foo//bar///baz");
		assert!(path_with_slashes.has_prefix("foo/bar"));
		assert_eq!(path_with_slashes.strip_prefix("foo").unwrap(), "bar/baz");
		assert_eq!(path_with_slashes.join("qux"), "foo/bar/baz/qux");

		// Test PathRef to Path conversion
		let path_ref = Path::new("foo//bar///baz");
		assert_eq!(path_ref, "foo/bar/baz"); // PathRef now normalizes too
		let path_from_ref = path_ref.to_owned();
		assert_eq!(path_from_ref, "foo/bar/baz"); // Both are normalized
	}

	#[test]
	fn test_path_ref_multiple_slashes() {
		// PathRef now normalizes multiple slashes using Cow
		let path_ref = Path::new("//foo//bar///baz//");
		assert_eq!(path_ref, "foo/bar/baz"); // Fully normalized

		// Various multiple slash scenarios are normalized in PathRef
		assert_eq!(Path::new("foo//bar"), "foo/bar");
		assert_eq!(Path::new("foo///bar"), "foo/bar");
		assert_eq!(Path::new("a//b//c//d"), "a/b/c/d");

		// Conversion to Path maintains normalized form
		assert_eq!(Path::new("foo//bar").to_owned(), "foo/bar");
		assert_eq!(Path::new("foo///bar").to_owned(), "foo/bar");
		assert_eq!(Path::new("a//b//c//d").to_owned(), "a/b/c/d");

		// Edge cases
		assert_eq!(Path::new("//"), "");
		assert_eq!(Path::new("////"), "");
		assert_eq!(Path::new("//").to_owned(), "");
		assert_eq!(Path::new("////").to_owned(), "");

		// Test that PathRef avoids allocation when no normalization needed
		let normal_path = Path::new("foo/bar/baz");
		assert_eq!(normal_path, "foo/bar/baz");
		// This should use Cow::Borrowed internally (no allocation)

		let needs_norm = Path::new("foo//bar");
		assert_eq!(needs_norm, "foo/bar");
		// This should use Cow::Owned internally (allocation only when needed)
	}

	#[test]
	fn test_ergonomic_conversions() {
		// Test that all these work ergonomically in function calls
		fn takes_path_ref<'a>(p: impl Into<Path<'a>>) -> String {
			p.into().to_string()
		}

		// Alternative API using the trait alias for better error messages
		fn takes_path_ref_with_trait<'a>(p: impl Into<Path<'a>>) -> String {
			p.into().to_string()
		}

		// String literal
		assert_eq!(takes_path_ref("foo//bar"), "foo/bar");

		// String (owned) - this should now work without &
		let owned_string = String::from("foo//bar///baz");
		assert_eq!(takes_path_ref(owned_string), "foo/bar/baz");

		// &String
		let string_ref = String::from("foo//bar");
		assert_eq!(takes_path_ref(string_ref), "foo/bar");

		// PathRef
		let path_ref = Path::new("foo//bar");
		assert_eq!(takes_path_ref(path_ref), "foo/bar");

		// Path
		let path = Path::new("foo//bar");
		assert_eq!(takes_path_ref(path), "foo/bar");

		// Test that Path::new works with all these types
		let _path1 = Path::new("foo/bar"); // &str
		let _path2 = Path::new("foo/bar"); // String - should now work
		let _path3 = Path::new("foo/bar"); // &String
		let _path4 = Path::new("foo/bar"); // PathRef

		// Test the trait alias version works the same
		assert_eq!(takes_path_ref_with_trait("foo//bar"), "foo/bar");
		assert_eq!(takes_path_ref_with_trait(String::from("foo//bar")), "foo/bar");
	}

	#[test]
	fn test_prefix_strip_prefix() {
		// Test basic stripping
		let prefix = Path::new("foo/bar/baz");
		assert_eq!(prefix.strip_prefix("").unwrap(), "foo/bar/baz");
		assert_eq!(prefix.strip_prefix("foo").unwrap(), "bar/baz");
		assert_eq!(prefix.strip_prefix("foo/").unwrap(), "bar/baz");
		assert_eq!(prefix.strip_prefix("foo/bar").unwrap(), "baz");
		assert_eq!(prefix.strip_prefix("foo/bar/").unwrap(), "baz");
		assert_eq!(prefix.strip_prefix("foo/bar/baz").unwrap(), "");

		// Test invalid prefixes
		assert!(prefix.strip_prefix("fo").is_none());
		assert!(prefix.strip_prefix("bar").is_none());
		assert!(prefix.strip_prefix("foo/ba").is_none());

		// Test edge cases
		let prefix = Path::new("foobar");
		assert!(prefix.strip_prefix("foo").is_none());
		assert_eq!(prefix.strip_prefix("foobar").unwrap(), "");

		// Test empty prefix
		let prefix = Path::new("");
		assert_eq!(prefix.strip_prefix("").unwrap(), "");
		assert!(prefix.strip_prefix("foo").is_none());

		// Test single component
		let prefix = Path::new("foo");
		assert_eq!(prefix.strip_prefix("foo").unwrap(), "");
		assert_eq!(prefix.strip_prefix("foo/").unwrap(), ""); // "foo/" becomes "foo" after trimming

		// Test trailing slash handling
		let prefix = Path::new("foo/bar/");
		assert_eq!(prefix.strip_prefix("foo").unwrap(), "bar");
		assert_eq!(prefix.strip_prefix("foo/").unwrap(), "bar");
		assert_eq!(prefix.strip_prefix("foo/bar").unwrap(), "");
		assert_eq!(prefix.strip_prefix("foo/bar/").unwrap(), "");
	}

	#[test]
	fn test_prefix_list_dedup() {
		// Exact duplicates are removed
		let list = PathPrefixes::new(["demo", "demo"]);
		assert_eq!(list.len(), 1);
		assert_eq!(list[0], Path::new("demo"));
	}

	#[test]
	fn test_prefix_list_overlap() {
		// "demo/foo" is redundant when "demo" exists
		let list = PathPrefixes::new(["demo", "demo/foo", "anon"]);
		assert_eq!(list.len(), 2);
		assert!(list.iter().any(|p| p == &Path::new("demo")));
		assert!(list.iter().any(|p| p == &Path::new("anon")));
	}

	#[test]
	fn test_prefix_list_overlap_reverse_order() {
		// Order shouldn't matter
		let list = PathPrefixes::new(["demo/foo", "demo"]);
		assert_eq!(list.len(), 1);
		assert_eq!(list[0], Path::new("demo"));
	}

	#[test]
	fn test_prefix_list_empty_covers_all() {
		// Empty prefix covers everything
		let list = PathPrefixes::new(["", "demo", "anon"]);
		assert_eq!(list.len(), 1);
		assert_eq!(list[0], Path::new(""));
	}

	#[test]
	fn test_prefix_list_no_overlap() {
		// Unrelated prefixes are all kept
		let list = PathPrefixes::new(["demo", "anon", "secret"]);
		assert_eq!(list.len(), 3);
	}

	#[test]
	fn test_prefix_list_single() {
		let list = PathPrefixes::new(["demo"]);
		assert_eq!(list.len(), 1);
	}

	#[test]
	fn test_prefix_list_empty() {
		let list = PathPrefixes::new(std::iter::empty::<&str>());
		assert!(list.is_empty());
		assert_eq!(list.len(), 0);
	}

	#[test]
	fn test_prefix_list_deep_overlap() {
		// "a/b/c" is covered by "a/b" which is covered by "a"
		let list = PathPrefixes::new(["a/b/c", "a/b", "a"]);
		assert_eq!(list.len(), 1);
		assert_eq!(list[0], Path::new("a"));
	}

	#[test]
	fn test_prefix_list_partial_name_not_overlap() {
		// "demo" should NOT cover "demonstration" (different path component)
		let list = PathPrefixes::new(["demo", "demonstration"]);
		assert_eq!(list.len(), 2);
	}

	#[test]
	fn test_prefix_list_collect() {
		let paths: Vec<PathOwned> = vec!["demo".into(), "demo/foo".into()];
		let list: PathPrefixes = paths.into_iter().collect();
		assert_eq!(list.len(), 1);
		assert_eq!(list[0], Path::new("demo"));
	}

	#[test]
	fn test_prefix_list_eq_vec() {
		let list = PathPrefixes::new(["demo", "anon"]);
		// Canonical order: sorted by length, then lexicographically
		assert_eq!(list, vec!["anon".as_path(), "demo".as_path()]);
	}

	// Pointer of the nth part, for asserting parts are shared rather than copied.
	fn part_ptr(path: &Path, n: usize) -> *const u8 {
		path.part(n).expect("part in range").as_ptr()
	}

	// Pointer-equality checks that owned paths share their part allocations through
	// the clone / to_owned / strip_prefix / join flows used by origin announce fan-out.
	#[test]
	fn test_owned_paths_share_allocation() {
		let path = Path::new("customer/room/broadcast").to_owned();

		// Cloning an owned path shares the parts.
		let cloned = path.clone();
		assert_eq!(part_ptr(&path, 0), part_ptr(&cloned, 0));

		// as_path + to_owned (how notify queues a path per consumer) shares too.
		let requeued = path.as_path().to_owned();
		assert_eq!(part_ptr(&path, 0), part_ptr(&requeued, 0));

		// Stripping a prefix from an owned path is offset arithmetic, not a copy.
		let stripped = path.strip_prefix("customer").unwrap().to_owned();
		assert_eq!(stripped, "room/broadcast");
		assert_eq!(part_ptr(&stripped, 0), part_ptr(&path, 1));

		// next_part shares the rest as well.
		let (dir, rest) = path.next_part().unwrap();
		assert_eq!(dir, "customer");
		let rest = rest.to_owned();
		assert_eq!(part_ptr(&rest, 0), part_ptr(&path, 1));

		// join shares the parts of both owned inputs; this is the filesystem-like
		// pattern where every path under a customer shares the prefix allocations.
		let joined = path.join(&stripped);
		assert_eq!(joined, "customer/room/broadcast/room/broadcast");
		assert_eq!(part_ptr(&joined, 0), part_ptr(&path, 0));
		assert_eq!(part_ptr(&joined, 3), part_ptr(&stripped, 0));
	}

	#[test]
	fn test_parts() {
		assert_eq!(Path::empty().parts().count(), 0);
		assert_eq!(Path::new("foo").parts().collect::<Vec<_>>(), ["foo"]);
		assert_eq!(Path::new("/foo//bar/").parts().collect::<Vec<_>>(), ["foo", "bar"]);
	}

	#[test]
	fn test_wire_max_parts() {
		use crate::lite::Version;

		let ok = (0..Path::MAX_PARTS)
			.map(|i| i.to_string())
			.collect::<Vec<_>>()
			.join("/");
		let too_deep = format!("{ok}/extra");

		// Encode enforces the limit.
		let mut buf = bytes::BytesMut::new();
		Path::new(&ok).encode(&mut buf, Version::Lite04).unwrap();
		assert!(matches!(
			Path::new(&too_deep).encode(&mut bytes::BytesMut::new(), Version::Lite04),
			Err(EncodeError::BoundsExceeded)
		));

		// Decode round-trips at the limit.
		let decoded = Path::decode(&mut buf.freeze(), Version::Lite04).unwrap();
		assert_eq!(decoded, ok.as_str());

		// Decode enforces the limit on a raw string that encode would have refused.
		let mut buf = bytes::BytesMut::new();
		too_deep.as_str().encode(&mut buf, Version::Lite04).unwrap();
		assert!(matches!(
			Path::decode(&mut buf.freeze(), Version::Lite04),
			Err(DecodeError::BoundsExceeded)
		));
	}

	#[test]
	fn test_owned_empty_paths() {
		// Empty paths never allocate and stay well-behaved.
		let empty = Path::new("").to_owned();
		assert!(empty.is_empty());
		assert_eq!(empty, Path::empty());

		let path = Path::new("foo").to_owned();
		let rest = path.strip_prefix("foo").unwrap().to_owned();
		assert!(rest.is_empty());
	}

	#[test]
	fn test_prefix_list_canonical_order() {
		// Same inputs in different order produce identical results
		let a = PathPrefixes::new(["foo", "bar"]);
		let b = PathPrefixes::new(["bar", "foo"]);
		assert_eq!(a, b);
	}
}
