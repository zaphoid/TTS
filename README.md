Installation:

Save the code to a Python file.

Install dependencies: pip install openai python-docx tk

Also install FFMPEG, choco is the easiest way

Make sure your OpenAI API key is set as an environment variable <----------

Run the application and follow the interface to:

Select your text file
Choose an output directory
Configure voices and style instructions
Preview the segmentation
Convert to speech


Key Features

This will use OpenAI to scan text and divide into narration, speech, code and change voice and any instructions. It overcomes the file size limit by 
outputting small chunks then reassembling them into one file as the last step.

Segment-Specific Style Instructions:

Each segment type (narrator, dialog, computer) can have its own unique style instructions
The application uses a tabbed interface to manage different style settings for each type
Appropriate instructions are applied to each segment during conversion


Tailored Style Presets:

Different preset suggestions for each segment type
Narrator presets focus on storytelling qualities
Dialog presets enhance conversational elements
Computer presets improve technical speech clarity


Improved Preview:

Preview now shows segment type, voice, and style instruction for each segment
Provides a better visualization of how the final audio will be structured


Enhanced Configuration Management:

Better organization of style instructions
Clearer logging of which instructions are being applied to which segments



How It Works
The application now processes your text in these steps:

Analyzes text to identify different segment types (narrative, dialog, code/computer)
Assigns appropriate voices to each segment based on your settings
Applies segment-specific style instructions to each segment
Processes each segment with the correct voice and style combination
Combines all segments into a final audio file

This creates a much more dynamic and engaging narration where each type of content is spoken in a way that best suits its nature. For example:

Narrative passages can be spoken in an authoritative or mysterious tone
Dialog can have natural conversational inflection
Code and terminal outputs can be spoken with technical precision

This implementation should provide significantly better results for your cyberpunk text, with its mix of narrative, dialog, and technical elements.
