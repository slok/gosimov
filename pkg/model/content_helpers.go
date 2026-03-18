package model

// NewContentText creates a text content part.
func NewContentText(text string) ContentPart {
	return ContentPart{Type: ContentPartTypeText, Text: text}
}

// NewContentImage creates an image content part.
func NewContentImage(data []byte, mimeType string) ContentPart {
	return ContentPart{Type: ContentPartTypeImage, Image: &ImageData{Data: data, MimeType: mimeType}}
}
