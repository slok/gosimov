package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	"github.com/invopop/jsonschema"
)

func FromType[T any]() (json.RawMessage, error) {
	t, err := typeOf[T]()
	if err != nil {
		return nil, err
	}

	s := reflector.ReflectFromType(t)
	b, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("marshal schema: %w", err)
	}

	return json.RawMessage(b), nil
}

func MustFromType[T any]() json.RawMessage {
	s, err := FromType[T]()
	if err != nil {
		panic(fmt.Sprintf("generate JSON schema: %v", err))
	}

	return s
}

func DecodeStrict(data json.RawMessage, dst any) error {
	if len(data) == 0 {
		return nil
	}

	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()

	if err := dec.Decode(dst); err != nil {
		return err
	}

	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("unexpected trailing data")
		}

		return err
	}

	return nil
}

func typeOf[T any]() (reflect.Type, error) {
	var zero T
	t := reflect.TypeOf(zero)
	if t == nil {
		return nil, fmt.Errorf("invalid nil type")
	}

	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("type must be a struct")
	}

	return t, nil
}

var reflector = jsonschema.Reflector{
	Anonymous:                  true,
	AssignAnchor:               false,
	DoNotReference:             true,
	ExpandedStruct:             true,
	RequiredFromJSONSchemaTags: true,
	AllowAdditionalProperties:  false,
}
