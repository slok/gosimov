package model_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/slok/gosimov/pkg/model"
)

func TestNewContentText(t *testing.T) {
	tests := map[string]struct {
		text string
	}{
		"Creating a text content part should keep the text.": {
			text: "hello",
		},

		"Creating a text content part should allow empty text.": {
			text: "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := model.NewContentText(test.text)

			assert.Equal(model.ContentPartTypeText, got.Type)
			assert.Equal(test.text, got.Text)
			assert.Nil(got.Image)
		})
	}
}

func TestNewContentImage(t *testing.T) {
	tests := map[string]struct {
		data     []byte
		mimeType string
	}{
		"Creating an image content part should keep image data and mime type.": {
			data:     []byte("pngdata"),
			mimeType: "image/png",
		},

		"Creating an image content part should allow empty values.": {
			data:     nil,
			mimeType: "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)

			got := model.NewContentImage(test.data, test.mimeType)

			assert.Equal(model.ContentPartTypeImage, got.Type)
			assert.Equal("", got.Text)
			if assert.NotNil(got.Image) {
				assert.Equal(test.data, got.Image.Data)
				assert.Equal(test.mimeType, got.Image.MimeType)
			}
		})
	}
}
