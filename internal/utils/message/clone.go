package message

import "github.com/slok/gosimov/pkg/model"

// CloneMessages returns a deep copy of messages.
func CloneMessages(messages []model.Message) []model.Message {
	if len(messages) == 0 {
		return nil
	}

	cloned := make([]model.Message, len(messages))
	for i := range messages {
		cloned[i] = CloneMessage(messages[i])
	}

	return cloned
}

// CloneMessage returns a deep copy of a message.
func CloneMessage(message model.Message) model.Message {
	cloned := message

	if len(message.Content) > 0 {
		cloned.Content = make([]model.ContentPart, len(message.Content))
		copy(cloned.Content, message.Content)

		for i := range cloned.Content {
			if cloned.Content[i].Image == nil {
				continue
			}

			image := *cloned.Content[i].Image
			if len(image.Data) > 0 {
				image.Data = append([]byte(nil), image.Data...)
			}

			cloned.Content[i].Image = &image
		}
	}

	if len(message.ToolCallRequests) > 0 {
		cloned.ToolCallRequests = make([]model.ToolCallRequest, len(message.ToolCallRequests))
		copy(cloned.ToolCallRequests, message.ToolCallRequests)

		for i := range cloned.ToolCallRequests {
			if len(cloned.ToolCallRequests[i].Arguments) == 0 {
				continue
			}

			cloned.ToolCallRequests[i].Arguments = append([]byte(nil), cloned.ToolCallRequests[i].Arguments...)
		}
	}

	if message.Metadata != nil {
		metadata := *message.Metadata
		if message.Metadata.Usage != nil {
			usage := *message.Metadata.Usage
			metadata.Usage = &usage
		}

		cloned.Metadata = &metadata
	}

	if message.Compaction != nil {
		compaction := *message.Compaction
		cloned.Compaction = &compaction
	}

	return cloned
}
