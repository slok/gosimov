package file_test

import (
	"bytes"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/internal/utils/file"
)

func TestDetectContent(t *testing.T) {
	tests := map[string]struct {
		data    []byte
		expKind file.ContentKind
		expMime string
	}{
		"Plain text should be detected as text.": {
			data:    []byte("hello world\nthis is text"),
			expKind: file.ContentKindText,
		},

		"Empty content should be detected as text.": {
			data:    []byte{},
			expKind: file.ContentKindText,
		},

		"PNG image should be detected.": {
			data:    encodePNG(t),
			expKind: file.ContentKindImage,
			expMime: "image/png",
		},

		"JPEG image should be detected.": {
			data:    encodeJPEG(t),
			expKind: file.ContentKindImage,
			expMime: "image/jpeg",
		},

		"GIF89a image should be detected.": {
			data:    encodeGIF(t),
			expKind: file.ContentKindImage,
			expMime: "image/gif",
		},

		"WebP image should be detected.": {
			data:    fakeWebP(),
			expKind: file.ContentKindImage,
			expMime: "image/webp",
		},

		"Binary with null bytes should be detected as binary.": {
			data:    []byte{0x01, 0x02, 0x00, 0x03, 0x04},
			expKind: file.ContentKindBinary,
		},

		"Compiled binary (ELF header) should be detected as binary.": {
			data:    []byte{0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00},
			expKind: file.ContentKindBinary,
		},

		"UTF-8 text with multibyte characters should be text.": {
			data:    []byte("こんにちは世界\néàü\n🚀"),
			expKind: file.ContentKindText,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			result := file.DetectContent(test.data)

			assert.Equal(t, test.expKind, result.Kind)
			assert.Equal(t, test.expMime, result.MimeType)
		})
	}
}

// encodePNG creates a minimal valid PNG image.
func encodePNG(t *testing.T) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 1, 1))

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("failed to encode PNG: %v", err)
	}

	return buf.Bytes()
}

// encodeJPEG creates a minimal valid JPEG image.
func encodeJPEG(t *testing.T) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 1, 1))

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatalf("failed to encode JPEG: %v", err)
	}

	return buf.Bytes()
}

// encodeGIF creates a minimal valid GIF image.
func encodeGIF(t *testing.T) []byte {
	t.Helper()

	img := image.NewPaletted(image.Rect(0, 0, 1, 1), color.Palette{color.Black})

	var buf bytes.Buffer
	if err := gif.Encode(&buf, img, nil); err != nil {
		t.Fatalf("failed to encode GIF: %v", err)
	}

	return buf.Bytes()
}

// fakeWebP creates a minimal byte sequence with the WebP magic signature.
func fakeWebP() []byte {
	// RIFF....WEBP — 12 bytes minimum.
	data := make([]byte, 20)
	copy(data[0:4], "RIFF")
	copy(data[8:12], "WEBP")

	return data
}
