package schema_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/slok/gosimov/pkg/tool/schema"
)

func TestFromType(t *testing.T) {
	t.Run("Struct with tags should generate strict schema.", func(t *testing.T) {
		assert := assert.New(t)
		require := require.New(t)

		type input struct {
			Path   string            `json:"path" jsonschema:"required,description=File path"`
			Labels map[string]string `json:"labels,omitempty" jsonschema:"description=Labels map"`
			Items  []struct {
				Count int `json:"count" jsonschema:"required,description=Nested count"`
			} `json:"items,omitempty" jsonschema:"description=Nested items"`
		}

		raw, err := schema.FromType[input]()
		require.NoError(err)
		assert.True(json.Valid(raw))

		var got map[string]any
		require.NoError(json.Unmarshal(raw, &got))

		assert.Equal("object", got["type"])
		assert.Equal(false, got["additionalProperties"])

		required, ok := got["required"].([]any)
		require.True(ok)
		assert.ElementsMatch([]any{"path"}, required)

		props, ok := got["properties"].(map[string]any)
		require.True(ok)

		path, ok := props["path"].(map[string]any)
		require.True(ok)
		assert.Equal("string", path["type"])
		assert.Equal("File path", path["description"])

		labels, ok := props["labels"].(map[string]any)
		require.True(ok)
		assert.Equal("object", labels["type"])

		items, ok := props["items"].(map[string]any)
		require.True(ok)
		assert.Equal("array", items["type"])

		itemSchema, ok := items["items"].(map[string]any)
		require.True(ok)
		assert.Equal("object", itemSchema["type"])
		assert.Equal(false, itemSchema["additionalProperties"])
	})

	t.Run("Non-struct type should fail.", func(t *testing.T) {
		assert := assert.New(t)
		require := require.New(t)

		_, err := schema.FromType[int]()
		require.Error(err)
		assert.Contains(err.Error(), "type must be a struct")
	})
}

func TestDecodeStrict(t *testing.T) {
	tests := map[string]struct {
		data      json.RawMessage
		expErr    bool
		errSubstr string
	}{
		"Valid payload should decode.": {
			data: json.RawMessage(`{"path":"a.txt","limit":10}`),
		},

		"Unknown field should fail.": {
			data:      json.RawMessage(`{"path":"a.txt","unknown":true}`),
			expErr:    true,
			errSubstr: "unknown field",
		},

		"Trailing data should fail.": {
			data:      json.RawMessage(`{"path":"a.txt"} {"x":1}`),
			expErr:    true,
			errSubstr: "unexpected trailing data",
		},

		"Empty payload should keep zero values.": {
			data: nil,
		},
	}

	type input struct {
		Path  string `json:"path"`
		Limit int    `json:"limit"`
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			assert := assert.New(t)
			require := require.New(t)

			var got input
			err := schema.DecodeStrict(test.data, &got)

			if test.expErr {
				require.Error(err)
				if test.errSubstr != "" {
					assert.Contains(err.Error(), test.errSubstr)
				}
				return
			}

			require.NoError(err)
		})
	}
}
