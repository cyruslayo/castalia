import json
import re

with open('EduTutor/04_agentic_tutor_ui.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'import gradio as gr' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # 1. Modify chat_fn to also return a mock scenario
        source = source.replace('return response, dashboard_text', 
'''    scenario_str = ""
    if "fraction" in message.lower() or "math" in subject.lower():
        scenario_str = json.dumps({
            "title": "Equivalent Fractions",
            "concept": "Fractions on a Number Line",
            "scaffolding_hints": ["Look at how 1/4 and 2/4 compare."],
            "playground_state": { "tool": "fraction_bars" }
        })
    return response, dashboard_text, scenario_str''')
            
        # 2. Add base64 encoding and iframe reading
        ui_code_start = '# Build the UI'
        new_ui_code_start = '''import base64
with open("math_playground.html", "r", encoding="utf-8") as f:
    html_content = f.read()
encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
iframe_src = f"data:text/html;base64,{encoded_html}"

# Build the UI'''
        source = source.replace(ui_code_start, new_ui_code_start)
        
        # 3. Insert iframe into UI
        col_start = '        with gr.Column(scale=3):'
        new_col_start = '''        with gr.Column(scale=3):
            gr.HTML(f'<iframe src="{iframe_src}" width="100%" height="400px" style="border: none; border-radius: 12px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></iframe>')
            scenario_json_box = gr.Textbox(visible=False, elem_id="scenario_json")'''
        source = source.replace(col_start, new_col_start)
        
        # 4. Update respond function
        new_respond = '''    def respond(message, history, profile, subject, topic):
        response, dashboard_text, scenario_str = chat_fn(message, history, profile, subject, topic)
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, "", dashboard_text, scenario_str'''
        
        source = re.sub(r'    def respond\(message, history, profile, subject, topic\):.*?return history, "", dashboard_text', new_respond, source, flags=re.DOTALL)
        
        # 5. Update event bindings
        old_send_click = '''    send_btn.click(
        respond,
        inputs=[msg, chatbot, profile_dropdown, subject_dropdown, topic_input],
        outputs=[chatbot, msg, dashboard_display],
    )'''
        new_send_click = '''    update_iframe_js = """
    function(json_str) {
        if (json_str) {
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                try {
                    const payload = JSON.parse(json_str);
                    iframe.contentWindow.postMessage({ type: 'SCENARIO_UPDATE', payload }, '*');
                } catch (e) {}
            });
        }
        return json_str;
    }
    """
    
    send_btn.click(
        respond,
        inputs=[msg, chatbot, profile_dropdown, subject_dropdown, topic_input],
        outputs=[chatbot, msg, dashboard_display, scenario_json_box],
    ).then(
        fn=None,
        inputs=[scenario_json_box],
        outputs=None,
        js=update_iframe_js
    )'''
        source = source.replace(old_send_click, new_send_click)
        
        old_msg_submit = '''    msg.submit(
        respond,
        inputs=[msg, chatbot, profile_dropdown, subject_dropdown, topic_input],
        outputs=[chatbot, msg, dashboard_display],
    )'''
        new_msg_submit = '''    msg.submit(
        respond,
        inputs=[msg, chatbot, profile_dropdown, subject_dropdown, topic_input],
        outputs=[chatbot, msg, dashboard_display, scenario_json_box],
    ).then(
        fn=None,
        inputs=[scenario_json_box],
        outputs=None,
        js=update_iframe_js
    )'''
        source = source.replace(old_msg_submit, new_msg_submit)
        
        # Break source back into lines
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
        break

with open('EduTutor/04_agentic_tutor_ui.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
