import uuid


def html_format(source, language, css_class, options, md):
    div_id = f"altair-plot-{uuid.uuid4()}"
    return f"""
        <div id="{ div_id }">
        <script>
          // embed when document is loaded, to ensure vega library is available
          // this works on all modern browsers, except IE8 and older
          document.addEventListener("DOMContentLoaded", function(event) {{
              var opt = {{
                "mode": "vega-lite",
                "renderer": "canvas",
                "actions": false,
              }};
              vegaEmbed('#{ div_id }', '{source}', opt).catch(console.err);
          }}, {{passive: true, once: true}});
        </script>
        </div>
        """
