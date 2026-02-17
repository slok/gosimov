package file

import (
	"io/fs"
	"os"
	"path/filepath"
)

// ReadWriteFS abstracts filesystem operations for components that read and write files.
//
// Go's [fs.FS] is read-only, so components that modify the filesystem need this interface.
// The default implementation ([NewOSReadWriteFS]) wraps [os] functions rooted at a directory.
// Tests can provide in-memory implementations.
//
// Consumers should define their own narrower interfaces if they don't need all methods.
type ReadWriteFS interface {
	// Stat returns file info, or an error if the file doesn't exist.
	Stat(path string) (fs.FileInfo, error)
	// ReadFile reads and returns the contents of the named file.
	ReadFile(path string) ([]byte, error)
	// ReadDir reads the named directory and returns its entries.
	ReadDir(path string) ([]fs.DirEntry, error)
	// MkdirAll creates a directory and all parents.
	MkdirAll(path string) error
	// WriteFile writes data to the named file, creating it if necessary.
	WriteFile(path string, data []byte) error
	// AppendFile appends data to the named file.
	AppendFile(path string, data []byte) error
}

// NewOSReadWriteFS returns a [ReadWriteFS] backed by [os] functions rooted at root.
func NewOSReadWriteFS(root string) ReadWriteFS {
	return &osReadWriteFS{root: root}
}

type osReadWriteFS struct {
	root string
}

func (f *osReadWriteFS) Stat(path string) (fs.FileInfo, error) {
	return os.Stat(filepath.Join(f.root, path))
}

func (f *osReadWriteFS) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(filepath.Join(f.root, path))
}

func (f *osReadWriteFS) MkdirAll(path string) error {
	return os.MkdirAll(filepath.Join(f.root, path), 0o755)
}

func (f *osReadWriteFS) WriteFile(path string, data []byte) error {
	return os.WriteFile(filepath.Join(f.root, path), data, 0o644)
}

func (f *osReadWriteFS) ReadDir(path string) ([]fs.DirEntry, error) {
	return os.ReadDir(filepath.Join(f.root, path))
}

func (f *osReadWriteFS) AppendFile(path string, data []byte) error {
	file, err := os.OpenFile(filepath.Join(f.root, path), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = file.Write(data)
	return err
}
