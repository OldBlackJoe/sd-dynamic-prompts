/* global titles:true */
// Mouseover tooltips for various UI elements.
// `titles` is already defined by A1111, so we just merge into it...
titles = {
  ...titles,
  "Conditional Dynamic Prompts enabled":
    "Disable conditional dynamic prompts by unchecking this box.",

  "Combinatorial generation": `
Only content explicitly wrapped in @combination{...} is enumerated. Other wildcards and variants are selected randomly again for every generated prompt.
For example, '@combination{__swimsuit__}, __pose__' generates one prompt per swimsuit line while keeping __pose__ random.
Multiple @combination{...} blocks form a Cartesian product with each other.

The value of the 'Seed' field is only used for the first image. To change this, look for 'Fixed seed' in the 'Advanced options' section.`.trim(),

  "Max generations (0 = all combinations - the batch count value is ignored)": `
Limit the maximum number of marked combinations generated. 0 (default) generates all @combination{...} results.
`.trim(),

  "Combinatorial batches": `Repeat each marked @combination{...} result this many times. Repetitions are placed next to each other, and unwrapped wildcards and variants are randomly expanded again each time. When Seed is -1 and Fixed seed is off, every image also receives an independent random seed.`,

  "Magic prompt": `
Magic Prompt adds interesting modifiers to your prompt for a little bit of extra spice.
The first time you use it, the MagicPrompt model is downloaded so be patient.
If you're running low on VRAM, you might get a CUDA error.`.trim(),

  "Max magic prompt length":
    "Controls the maximum length in tokens of the generated prompt.",
  "Magic prompt creativity":
    "Adjusts the generated prompt. You will need to experiment with this setting.",
  "Magic Prompt batch size":
    "The number of prompts to generate per batch. Increasing this can speed up prompt generation at the expense of slightly increased VRAM usage.",

  "I'm feeling lucky": `
Uses the lexica.art API to create random prompts.
The prompt in the main prompt box is used as a search string.
Leaving the prompt box blank returns a list of completely randomly chosen prompts.
Try it out, it can be quite fun.
`.trim(),

  "Attention grabber": `Randomly selects a keyword from the prompt and adds emphasis to it. Try this with Fixed Seed enabled.`,

  "Write prompts to file": `
The generated file is a slugified version of the prompt and can be found in the same directory as the generated images.
E.g. in ./outputs/txt2img-images/.`.trim(),

  "Don't generate images":
    "Be sure to check the 'Write prompts to file' checkbox if you don't want to lose the generated prompts. Note, one image is still generated.",
  "Enable Jinja2 templates":
    "Jinja2 templates are an expressive alternative to the standard syntax. See the Help section below for instructions.",
  "Unlink seed from prompt":
    "Check this if you want to generate random prompts, even if your seed is fixed",
  "Don't apply to negative prompts":
    "Don't use prompt magic on negative prompts.",

  "Fixed seed": `
Select this if you want to use the same seed for every generated image.
This is useful if you want to test prompt variations while using the same seed.
If there are no wildcards then all the images will be identical.
`.trim(),
  "Write raw prompt to image":
    "Write the prompt template into the image metadata",
};
