package file

import "bytes"

// ContentKind describes the detected type of file content.
type ContentKind int

const (
	// ContentKindText is a text file.
	ContentKindText ContentKind = iota
	// ContentKindImage is a recognized image file.
	ContentKindImage
	// ContentKindBinary is a non-text, non-image binary file.
	ContentKindBinary
)

// sniffSize is the number of bytes read from the start of a file to detect its type.
const sniffSize = 4096

// imageMagic maps image MIME types to their magic byte signatures.
var imageMagic = []struct {
	mime   string
	offset int
	magic  []byte
}{
	{mime: "image/png", offset: 0, magic: []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}}, // \x89PNG\r\n\x1a\n
	{mime: "image/jpeg", offset: 0, magic: []byte{0xFF, 0xD8, 0xFF}},
	{mime: "image/gif", offset: 0, magic: []byte("GIF87a")},
	{mime: "image/gif", offset: 0, magic: []byte("GIF89a")},
	{mime: "image/webp", offset: 8, magic: []byte("WEBP")}, // RIFF....WEBP (offset 8).
}

// DetectResult holds the result of content detection.
type DetectResult struct {
	Kind     ContentKind
	MimeType string // Set only when Kind == ContentKindImage.
}

// DetectContent inspects raw file data and determines whether it is text,
// a recognized image, or binary.
//
// Image detection uses magic byte signatures. Binary detection checks for
// null bytes in the first [sniffSize] bytes.
func DetectContent(data []byte) DetectResult {
	// Check image magic bytes first.
	for _, m := range imageMagic {
		end := m.offset + len(m.magic)
		if len(data) >= end && bytes.Equal(data[m.offset:end], m.magic) {
			return DetectResult{Kind: ContentKindImage, MimeType: m.mime}
		}
	}

	// Check for binary content (null bytes in the sniff window).
	window := data
	if len(window) > sniffSize {
		window = window[:sniffSize]
	}

	if bytes.ContainsRune(window, 0) {
		return DetectResult{Kind: ContentKindBinary}
	}

	return DetectResult{Kind: ContentKindText}
}
