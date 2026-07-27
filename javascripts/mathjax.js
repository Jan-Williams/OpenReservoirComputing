window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ['$', '$']],
    displayMath: [["\\[", "\\]"], ['$$', '$$']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    processHtmlClass: "jp-RenderedMarkdown|arithmatex"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      // Process HTML entities in math content
      const mathElements = document.querySelectorAll('.jp-RenderedMarkdown');
      mathElements.forEach(el => {
        el.innerHTML = el.innerHTML
          .replace(/&amp;/g, '&')
          .replace(/&lt;/g, '<')
          .replace(/&gt;/g, '>')
          .replace(/&quot;/g, '"');
      });
      MathJax.startup.document.render();
    }
  }
};
