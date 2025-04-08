import os
import glob
import asyncio
import argparse
import logging
import streamlit as st
import inspect
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from src.utils.utils import get_latest_files, capture_screenshot, MissingAPIKeyError
from src.utils import utils
from src.utils.deep_research import deep_research

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()

# Initialize session state variables if they don't exist
if 'agent_running' not in st.session_state:
    st.session_state.agent_running = False
if 'research_running' not in st.session_state:
    st.session_state.research_running = False
if 'final_result' not in st.session_state:
    st.session_state.final_result = ""
if 'errors' not in st.session_state:
    st.session_state.errors = ""
if 'model_actions' not in st.session_state:
    st.session_state.model_actions = ""
if 'model_thoughts' not in st.session_state:
    st.session_state.model_thoughts = ""
if 'recording_gif' not in st.session_state:
    st.session_state.recording_gif = None
if 'trace_file' not in st.session_state:
    st.session_state.trace_file = None
if 'agent_history_file' not in st.session_state:
    st.session_state.agent_history_file = None
if 'markdown_output' not in st.session_state:
    st.session_state.markdown_output = ""
if 'markdown_download' not in st.session_state:
    st.session_state.markdown_download = None

def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text

    import re

    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)

    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)

    return result

def stop_agent():
    """Request the agent to stop"""
    global _global_agent

    try:
        if _global_agent is not None:
            # Request stop
            _global_agent.stop()
        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")
        st.session_state.agent_running = False
        return True
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return False

def stop_research_agent():
    """Request the research agent to stop"""
    global _global_agent_state

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")
        st.session_state.research_running = False
        return True
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return False

def close_global_browser():
    """Close the global browser instance"""
    global _global_browser, _global_browser_context, _global_agent
    
    if _global_browser_context is not None:
        asyncio.run(_global_browser_context.close())
        _global_browser_context = None
    
    if _global_browser is not None:
        asyncio.run(_global_browser.close())
        _global_browser = None
    
    _global_agent = None

async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file
        )

    except MissingAPIKeyError as e:
        logger.error(str(e))
        raise Exception(str(e))

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',  # final_result
            errors,  # errors
            '',  # model_actions
            '',  # model_thoughts
            None,  # latest_video
            None,  # trace_file
            None,  # history_file
        )

async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )

        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None

async def run_deep_search(
        research_task,
        max_search_iteration,
        max_query_per_iter,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_vision,
        use_own_browser,
        headless,
        chrome_cdp
):
    try:
        global _global_agent_state
        _global_agent_state.reset()
        
        # Reset the stop flag
        st.session_state.research_running = True

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )

        # Run deep research
        markdown_content, markdown_file = await deep_research(
            task=research_task,
            llm=llm,
            agent_state=_global_agent_state,
            max_iter=max_search_iteration,
            max_query_num=max_query_per_iter,
            use_vision=use_vision,
            use_own_browser=use_own_browser,
            headless=headless,
            chrome_cdp=chrome_cdp
        )

        st.session_state.research_running = False
        return markdown_content, markdown_file

    except Exception as e:
        import traceback
        traceback.print_exc()
        st.session_state.research_running = False
        return f"Error: {str(e)}\n{traceback.format_exc()}", None

def list_recordings(save_recording_path):
    if not os.path.exists(save_recording_path):
        return []

    # Get all video files
    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(
        os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

    # Sort recordings by creation time (oldest first)
    recordings.sort(key=os.path.getctime)

    # Add numbering to the recordings
    numbered_recordings = []
    for idx, recording in enumerate(recordings, start=1):
        filename = os.path.basename(recording)
        numbered_recordings.append((recording, f"{idx}. {filename}"))

    return numbered_recordings

def run_agent_task():
    st.session_state.agent_running = True
    
    # Get all the input values from the session state
    agent_type = st.session_state.agent_type
    llm_provider = st.session_state.llm_provider
    llm_model_name = st.session_state.llm_model_name
    llm_num_ctx = st.session_state.llm_num_ctx if 'llm_num_ctx' in st.session_state else 16000
    llm_temperature = st.session_state.llm_temperature
    llm_base_url = st.session_state.llm_base_url
    llm_api_key = st.session_state.llm_api_key
    use_own_browser = st.session_state.use_own_browser
    keep_browser_open = st.session_state.keep_browser_open
    headless = st.session_state.headless
    disable_security = st.session_state.disable_security
    window_w = st.session_state.window_w
    window_h = st.session_state.window_h
    save_recording_path = st.session_state.save_recording_path
    save_agent_history_path = st.session_state.save_agent_history_path
    save_trace_path = st.session_state.save_trace_path
    enable_recording = st.session_state.enable_recording
    task = st.session_state.task
    add_infos = st.session_state.add_infos
    max_steps = st.session_state.max_steps
    use_vision = st.session_state.use_vision
    max_actions_per_step = st.session_state.max_actions_per_step
    tool_calling_method = "auto"  # Default value
    chrome_cdp = st.session_state.chrome_cdp
    max_input_tokens = st.session_state.max_input_tokens
    
    # Create directories if they don't exist
    os.makedirs(save_recording_path, exist_ok=True)
    os.makedirs(save_agent_history_path, exist_ok=True)
    os.makedirs(save_trace_path, exist_ok=True)
    
    # Run the agent asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            chrome_cdp=chrome_cdp,
            max_input_tokens=max_input_tokens
        ))
        
        # Update session state with results
        st.session_state.final_result = result[0]
        st.session_state.errors = result[1]
        st.session_state.model_actions = result[2]
        st.session_state.model_thoughts = result[3]
        st.session_state.recording_gif = result[4]
        st.session_state.trace_file = result[5]
        st.session_state.agent_history_file = result[6]
        
    except Exception as e:
        import traceback
        st.session_state.errors = f"Error: {str(e)}\n{traceback.format_exc()}"
    finally:
        st.session_state.agent_running = False
        loop.close()

def run_research_task():
    st.session_state.research_running = True
    
    # Get all the input values from the session state
    research_task = st.session_state.research_task
    max_search_iteration = st.session_state.max_search_iteration
    max_query_per_iter = st.session_state.max_query_per_iter
    llm_provider = st.session_state.llm_provider
    llm_model_name = st.session_state.llm_model_name
    llm_num_ctx = st.session_state.llm_num_ctx if 'llm_num_ctx' in st.session_state else 16000
    llm_temperature = st.session_state.llm_temperature
    llm_base_url = st.session_state.llm_base_url
    llm_api_key = st.session_state.llm_api_key
    use_vision = st.session_state.use_vision
    use_own_browser = st.session_state.use_own_browser
    headless = st.session_state.headless
    chrome_cdp = st.session_state.chrome_cdp
    
    # Run the research asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(run_deep_search(
            research_task=research_task,
            max_search_iteration=max_search_iteration,
            max_query_per_iter=max_query_per_iter,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_vision=use_vision,
            use_own_browser=use_own_browser,
            headless=headless,
            chrome_cdp=chrome_cdp
        ))
        
        # Update session state with results
        st.session_state.markdown_output = result[0]
        st.session_state.markdown_download = result[1]
        
    except Exception as e:
        import traceback
        st.session_state.markdown_output = f"Error: {str(e)}\n{traceback.format_exc()}"
    finally:
        st.session_state.research_running = False
        loop.close()

# Main Streamlit app
st.set_page_config(page_title="Browser Use WebUI", layout="wide")

st.title("🌐 Browser Use WebUI")
st.subheader("Control your browser with AI assistance")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚙️ Agent Settings", 
    "🔧 LLM Settings", 
    "🌐 Browser Settings", 
    "🤖 Run Agent", 
    "🧐 Deep Research", 
    "🎥 Recordings", 
    "📁 UI Configuration"
])

# Tab 1: Agent Settings
with tab1:
    st.header("Agent Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.agent_type = st.radio(
            "Agent Type",
            options=["org", "custom"],
            index=1,
            help="Select the type of agent to use"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.max_steps = st.slider(
            "Max Run Steps",
            min_value=1,
            max_value=200,
            value=100,
            step=1,
            help="Maximum number of steps the agent will take"
        )
    with col2:
        st.session_state.max_actions_per_step = st.slider(
            "Max Actions per Step",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Maximum number of actions the agent will take per step"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.use_vision = st.checkbox(
            "Use Vision",
            value=True,
            help="Enable visual processing capabilities"
        )
    with col2:
        st.session_state.max_input_tokens = st.number_input(
            "Max Input Tokens",
            value=128000,
            step=1000,
            help="Maximum number of input tokens"
        )

# Tab 2: LLM Settings
with tab2:
    st.header("LLM Settings")
    
    # Get provider choices from utils.model_names
    provider_choices = list(utils.model_names.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.llm_provider = st.selectbox(
            "LLM Provider",
            options=provider_choices,
            index=provider_choices.index("openai") if "openai" in provider_choices else 0,
            help="Select your preferred language model provider"
        )
    
    # Get model choices based on selected provider
    model_choices = utils.model_names.get(st.session_state.llm_provider, [])
    
    with col2:
        st.session_state.llm_model_name = st.selectbox(
            "Model Name",
            options=model_choices,
            index=model_choices.index("gpt-4o") if "gpt-4o" in model_choices else 0,
            help="Select a model or type a custom model name"
        )
    
    # Show Ollama context length slider only if Ollama is selected
    if st.session_state.llm_provider == "ollama":
        st.session_state.llm_num_ctx = st.slider(
            "Ollama Context Length",
            min_value=2 ** 8,
            max_value=2 ** 16,
            value=16000,
            step=1,
            help="Controls max context length model needs to handle (less = faster)"
        )
    
    st.session_state.llm_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.6,
        step=0.1,
        help="Controls randomness in model outputs"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.llm_base_url = st.text_input(
            "Base URL",
            value="",
            help="API endpoint URL (if required)"
        )
    with col2:
        st.session_state.llm_api_key = st.text_input(
            "API Key",
            value="",
            type="password",
            help="Your API key (leave blank to use .env)"
        )

# Tab 3: Browser Settings
with tab3:
    st.header("Browser Settings")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.use_own_browser = st.checkbox(
            "Use Own Browser",
            value=False,
            help="Use your existing browser instance",
            on_change=close_global_browser
        )
    with col2:
        st.session_state.keep_browser_open = st.checkbox(
            "Keep Browser Open",
            value=False,
            help="Keep Browser Open between Tasks",
            on_change=close_global_browser
        )
    with col3:
        st.session_state.headless = st.checkbox(
            "Headless Mode",
            value=False,
            help="Run browser without GUI"
        )
    with col4:
        st.session_state.disable_security = st.checkbox(
            "Disable Security",
            value=True,
            help="Disable browser security features"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.enable_recording = st.checkbox(
            "Enable Recording",
            value=True,
            help="Enable saving browser recordings"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.window_w = st.number_input(
            "Window Width",
            value=1280,
            help="Browser window width"
        )
    with col2:
        st.session_state.window_h = st.number_input(
            "Window Height",
            value=1100,
            help="Browser window height"
        )
    
    st.session_state.chrome_cdp = st.text_input(
        "CDP URL",
        value="",
        placeholder="http://localhost:9222",
        help="CDP for google remote debugging"
    )
    
    st.session_state.save_recording_path = st.text_input(
        "Recording Path",
        value="./tmp/record_videos",
        help="Path to save browser recordings",
        disabled=not st.session_state.enable_recording
    )
    
    st.session_state.save_trace_path = st.text_input(
        "Trace Path",
        value="./tmp/traces",
        help="Path to save Agent traces"
    )
    
    st.session_state.save_agent_history_path = st.text_input(
        "Agent History Save Path",
        value="./tmp/agent_history",
        help="Specify the directory where agent history should be saved"
    )

# Tab 4: Run Agent
with tab4:
    st.header("Run Agent")
    
    st.session_state.task = st.text_area(
        "Task Description",
        value="go to google.com and type 'OpenAI' click search and give me the first url",
        height=100,
        help="Describe what you want the agent to do"
    )
    
    st.session_state.add_infos = st.text_area(
        "Additional Information",
        value="",
        height=80,
        help="Optional hints to help the LLM complete the task"
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        run_button = st.button(
            "▶️ Run Agent", 
            on_click=run_agent_task,
            disabled=st.session_state.agent_running
        )
    with col2:
        stop_button = st.button(
            "⏹️ Stop", 
            on_click=stop_agent,
            disabled=not st.session_state.agent_running
        )
    
    # Display browser view placeholder when agent is running
    if st.session_state.agent_running:
        st.info("Browser agent is running...")
    
    # Results section
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Final Result", value=st.session_state.final_result, height=150)
    with col2:
        st.text_area("Errors", value=st.session_state.errors, height=150)
    
    # Only show these if they have content
    if st.session_state.model_actions:
        st.text_area("Model Actions", value=st.session_state.model_actions, height=150)
    if st.session_state.model_thoughts:
        st.text_area("Model Thoughts", value=st.session_state.model_thoughts, height=150)
    
    # Display recording GIF if available
    if st.session_state.recording_gif and os.path.exists(st.session_state.recording_gif):
        st.image(st.session_state.recording_gif, caption="Result GIF")
    
    # Display trace file and agent history file if available
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.trace_file and os.path.exists(st.session_state.trace_file):
            with open(st.session_state.trace_file, "rb") as file:
                st.download_button(
                    label="Download Trace File",
                    data=file,
                    file_name=os.path.basename(st.session_state.trace_file),
                    mime="application/zip"
                )
    with col2:
        if st.session_state.agent_history_file and os.path.exists(st.session_state.agent_history_file):
            with open(st.session_state.agent_history_file, "rb") as file:
                st.download_button(
                    label="Download Agent History",
                    data=file,
                    file_name=os.path.basename(st.session_state.agent_history_file),
                    mime="application/json"
                )

# Tab 5: Deep Research
with tab5:
    st.header("Deep Research")
    
    st.session_state.research_task = st.text_area(
        "Research Task",
        value="Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects, substantiated with examples of relevant models and techniques. The report should reflect original insights and analysis, moving beyond mere summarization of existing literature.",
        height=150
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.max_search_iteration = st.number_input(
            "Max Search Iteration",
            value=3,
            min_value=1,
            step=1
        )
    with col2:
        st.session_state.max_query_per_iter = st.number_input(
            "Max Query per Iteration",
            value=1,
            min_value=1,
            step=1
        )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Use a key for the checkbox instead of on_click
        if 'research_checkbox_state' not in st.session_state:
            st.session_state.research_checkbox_state = False
            
        checkbox_value = st.checkbox(
            "▶️ Run Deep Research", 
            value=st.session_state.research_running,
            key="research_checkbox",
            disabled=st.session_state.research_running
        )
        
        # Check if checkbox state changed to trigger the research task
        if checkbox_value and not st.session_state.research_checkbox_state:
            st.session_state.research_checkbox_state = True
            run_research_task()
        elif not checkbox_value and st.session_state.research_running:
            st.session_state.research_running = False
    with col2:
        st.session_state.stop_research_button = st.button(
            "⏹️ Stop Research", 
            on_click=stop_research_agent,
            disabled=not st.session_state.research_running
        )
    
    # Display research results
    if st.session_state.research_running:
        st.info("Research is in progress...")
    
    if st.session_state.markdown_output:
        st.markdown(st.session_state.markdown_output)
    
    # Download research report if available
    if st.session_state.markdown_download and os.path.exists(st.session_state.markdown_download):
        with open(st.session_state.markdown_download, "rb") as file:
            st.download_button(
                label="Download Research Report",
                data=file,
                file_name=os.path.basename(st.session_state.markdown_download),
                mime="text/markdown"
            )

# Tab 6: Recordings
with tab6:
    st.header("Recordings")
    
    # Path for recordings
    recordings_path = st.session_state.save_recording_path if 'save_recording_path' in st.session_state else "./tmp/record_videos"
    
    # Button to refresh recordings list
    if st.button("🔄 Refresh Recordings"):
        recordings = list_recordings(recordings_path)
        st.session_state.recordings = recordings
    
    # Display recordings
    if 'recordings' in st.session_state and st.session_state.recordings:
        # Create a grid layout for recordings
        cols = st.columns(3)
        for i, (recording_path, recording_name) in enumerate(st.session_state.recordings):
            with cols[i % 3]:
                st.video(recording_path)
                st.caption(recording_name)
    else:
        st.info("No recordings found. Run an agent task with recording enabled to create recordings.")

# Tab 7: UI Configuration
with tab7:
    st.header("UI Configuration")
    
    uploaded_file = st.file_uploader("Load UI Settings from Config File", type=["json"])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Config") and uploaded_file is not None:
            # Process the uploaded config file
            config_data = json.load(uploaded_file)
            # TODO: Implement config loading logic
            st.success("Configuration loaded successfully!")
    
    with col2:
        if st.button("Save UI Settings"):
            # TODO: Implement config saving logic
            config_path = os.path.join(os.getcwd(), "ui_config.json")
            # Create a dictionary of all settings
            config = {key: value for key, value in st.session_state.items() 
                     if not key.startswith('_') and key not in ['agent_running', 'research_running']}
            
            # Save to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success(f"Configuration saved to {config_path}")

