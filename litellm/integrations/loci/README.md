# litellm.integrations.loci

Compatibility path. These modules contain no logic: each re-exports the same names from the `loci_litellm`
package in `litellm-loci/`, so `litellm.integrations.loci.<module>` and `loci_litellm.<module>` are the same
objects. Importing this package adds `litellm-loci/` to `sys.path` and raises an `ImportError` naming the
problem if `loci_litellm` cannot be found.

What the callbacks do, every environment variable, and how failures show up are documented once, in
`litellm-loci/README.md`.
